"""A pretrained model as a zero-shot Mario player: describe the screen in text, ask for one button.

Used by eval_stages.py (--llm ollama:<model>) so a language model is scored with exactly the protocol the trained
agent and both baselines use. Nothing here trains anything.

Screen -> prompt. The 13x16 tile grid becomes a small text map (one character per tile) plus the physics and landing
hints in words, which is ~4x fewer tokens than the raw ids and far easier for a language model to read:
    .  empty      #  solid (ground, wall, brick, any other collidable tile)
    P  pipe       Q  ?-block      E  enemy      M  Mario      ?  not rendered yet (look-ahead only)
Only the tiles below are identified by id (all confirmed in this repo); every other non-zero tile is drawn as '#',
because the collision code treats any non-zero metatile as solid. A coin would therefore show as '#'.

Answer -> action. Ollama's structured outputs constrain the reply to a JSON object whose "action" is one of the
action names, so there is no free-text parsing and no silent fallback. A failed request is retried once; if it
fails again the evaluation stops with the error, rather than quietly substituting a move.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np

from env.tiles import ENEMY_BASE, MARIO_ID, UNKNOWN_ID, read_hints, read_landing_hints

PIPE_IDS = {0x12, 0x13, 0x14, 0x15}
QBLOCK_IDS = {0xC0, 0xC1, 0xC2}
WINDOWS_CURL = "/mnt/c/Windows/System32/curl.exe"

SYSTEM = (
    "You control Mario in Super Mario Bros. Every step you see the screen as a text map and choose one button "
    "combination, held for 4 frames. Goal: move right as far as possible and reach the flagpole without dying. "
    "Falling into a pit (a column with no ground) or touching an enemy from the side kills you. Jumping on an "
    "enemy from above defeats it. Holding A longer jumps higher and farther; B makes you run, which makes jumps "
    "longer. Map legend: '.' empty, '#' solid, 'P' pipe, 'Q' ?-block, 'E' enemy, 'M' Mario, '?' not visible yet. "
    "Rows go from the top of the screen (row 0) to the ground (row 12)."
)


def tile_char(v: int) -> str:
    if v == 0:
        return "."
    if v == MARIO_ID:
        return "M"
    if v == UNKNOWN_ID:
        return "?"
    if ENEMY_BASE <= v < MARIO_ID:
        return "E"
    if v in PIPE_IDS:
        return "P"
    if v in QBLOCK_IDS:
        return "Q"
    return "#"


def text_map(grid: np.ndarray) -> str:
    return "\n".join("".join(tile_char(int(v)) for v in row) for row in grid)


def describe_physics(ram) -> str:
    """The v4 physics hints and the landing hints, in words."""
    h = read_hints(ram)
    land = read_landing_hints(ram)
    speed = int(ram[0x57]) - 256 if int(ram[0x57]) > 127 else int(ram[0x57])
    airborne = int(ram[0x1D]) != 0
    parts = [f"Mario is {'in the air' if airborne else 'on the ground'}, horizontal speed {speed} "
             f"(28 = top walking speed, 48 = top running speed, negative = moving left)."]
    if h[0] >= 1.0:
        parts.append("No pit within 10 tiles ahead.")
    else:
        parts.append(f"Next pit starts {h[0] * 10:.1f} tiles ahead and is {h[1] * 10:.1f} tiles wide.")
        clears = [name for name, ok in zip(("a short walking jump", "a full walking jump", "a short running jump",
                                            "a full running jump"), h[3:7]) if ok]
        parts.append(f"It can be cleared by: {', '.join(clears)}." if clears else "No single jump clears it.")
    if land[3]:
        parts.append(f"A full jump taken now lands {land[1] * 10:.1f} tiles ahead, on "
                     f"{'solid ground' if land[0] else 'NOTHING (a pit)'}.")
    if land[2] > 0:
        parts.append(f"Warning: at this speed you walk off the edge in about {1 / land[2] - 1:.0f} steps.")
    if h[7] > 0:
        parts.append(f"An enemy ahead at your height will reach you in about {1 / h[7] - 1:.0f} steps.")
    if h[8] > 0:
        parts.append("You are falling onto an enemy (a stomp).")
    return " ".join(parts)


def build_prompt(grid: np.ndarray, ram, action_names: list[str], last_action: str | None) -> str:
    return (f"Screen:\n{text_map(grid)}\n\n{describe_physics(ram)}\n"
            f"Last button choice: {last_action or 'none'}.\n"
            f"Choose one of: {', '.join(action_names)}.")


class OllamaBackend:
    """Talks to a local Ollama server. From WSL, Windows' localhost is not reachable over the network, so when the
    direct request fails and Windows' curl.exe exists, requests go through it (WSL can run Windows programs, and
    they see Windows' localhost). Measured overhead of that route: ~0.025 s per call."""

    def __init__(self, model: str, url: str = "http://127.0.0.1:11434", num_ctx: int = 2048, seed: int = 0):
        self.model, self.url, self.num_ctx, self.seed = model, url.rstrip("/"), num_ctx, seed
        self.transport = self._pick_transport()
        self.latencies: list[float] = []

    def _pick_transport(self) -> str:
        try:
            urllib.request.urlopen(f"{self.url}/api/version", timeout=2).read()
            return "http"
        except (urllib.error.URLError, OSError):
            if os.path.exists(WINDOWS_CURL):
                out = subprocess.run([WINDOWS_CURL, "-s", "-m", "5", f"{self.url}/api/version"],
                                     capture_output=True, text=True)
                if out.returncode == 0 and "version" in out.stdout:
                    return "windows-curl"
        raise ConnectionError(f"no Ollama server reachable at {self.url} (directly or via Windows curl.exe)")

    def _request(self, path: str, payload: dict | None = None, timeout: int = 120) -> dict:
        body = json.dumps(payload).encode() if payload is not None else None
        if self.transport == "http":
            req = urllib.request.Request(f"{self.url}{path}", data=body, headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        cmd = [WINDOWS_CURL, "-s", "-m", str(timeout), f"{self.url}{path}"]
        if body is not None:
            cmd[4:4] = ["-H", "Content-Type: application/json", "--data-binary", "@-"]
        out = subprocess.run(cmd, input=body, capture_output=True)
        if out.returncode != 0:
            raise ConnectionError(f"curl.exe failed ({out.returncode}): {out.stderr.decode(errors='replace')}")
        return json.loads(out.stdout)

    def info(self) -> dict:
        """Model identity for the results file, so a later run can tell whether the model changed."""
        digest = next((m.get("digest") for m in self._request("/api/tags").get("models", [])
                       if m.get("name") == self.model), None)
        details = self._request("/api/show", {"model": self.model}).get("details", {})
        version = self._request("/api/version").get("version")
        return {"backend": "ollama", "model": self.model, "digest": digest, "details": details,
                "ollama_version": version, "transport": self.transport, "num_ctx": self.num_ctx, "seed": self.seed}

    def choose(self, prompt: str, action_names: list[str]) -> str:
        schema = {"type": "object", "properties": {"action": {"type": "string", "enum": action_names}},
                  "required": ["action"]}
        t0 = time.perf_counter()
        r = self._request("/api/generate", {
            "model": self.model, "system": SYSTEM, "prompt": prompt, "stream": False, "format": schema,
            "keep_alive": "10m", "options": {"temperature": 0, "seed": self.seed, "num_ctx": self.num_ctx}})
        self.latencies.append(time.perf_counter() - t0)
        action = json.loads(r["response"])["action"]
        if action not in action_names:
            raise ValueError(f"model returned {action!r}, not one of {action_names}")
        return action


class JevBackend:
    """Placeholder until early access: the request format is not public to us, so nothing is guessed here."""

    def __init__(self, *_, **__):
        raise NotImplementedError("Jev backend not written yet: add it once you have early access and its API "
                                  "docs; keep the same choose(prompt, action_names) -> name interface.")


def make_backend(spec: str):
    """'ollama:llama3.2:1b' -> OllamaBackend('llama3.2:1b'); 'jev' -> JevBackend."""
    kind, _, rest = spec.partition(":")
    if kind == "ollama":
        if not rest:
            raise ValueError("use ollama:<model>, e.g. ollama:llama3.2:1b")
        return OllamaBackend(rest)
    if kind == "jev":
        return JevBackend(rest)
    raise ValueError(f"unknown LLM backend {spec!r}; expected ollama:<model> or jev")


def action_labels(policy_action_names: list[str]) -> list[str]:
    """'RIGHT A B' -> 'RIGHT_A_B': single tokens are easier for the model and for the schema."""
    return [n.replace(" ", "_") for n in policy_action_names]
