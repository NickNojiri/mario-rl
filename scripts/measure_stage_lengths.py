"""Measure each stage's flagpole x_pos from the emulator, and write stages.json.

    python -m scripts.measure_stage_lengths                 # harvest recorded flag episodes -> stages.json
    python -m scripts.measure_stage_lengths --verify 1-1    # replay one in the emulator and confirm
    python -m scripts.measure_stage_lengths --verify all

Raw x_pos is not comparable across stages of different lengths, so reporting a mean over stages silently
weights the long ones. Dividing by the flagpole position fixes that, but only if the flagpole position is
something we measured rather than something we looked up.

Method. When an episode ends with flag_get, the x_pos recorded at that moment is read from the game's own RAM
(0x6D/0x86) and is the flagpole position. Every eval JSON under runs/ stores per-episode records, so this
harvests them and keeps a stage's value only when every recorded flag episode for it agrees. --verify replays
one of those episodes in the emulator from its checkpoint and seed (the run is deterministic, see
tests/test_determinism.py) and checks the value again end to end.

Stages the agent has never finished have no measurement and are written as null. They are placeholders to be
filled by a real run, never by an estimate or an internet value.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import time
from pathlib import Path

OUT = Path("stages.json")
ABOUT = ("flagpole x_pos per stage, measured from the emulator. null = never reached, so not measured. "
         "Never fill these from published level data; run the game.")


def harvest(pattern: str = "runs/*/eval*.json") -> dict[str, dict]:
    """Collect x_pos at flag_get from every recorded eval episode, grouped by stage."""
    found: dict[str, list] = collections.defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in data.get("stages", []):
            for ep in row.get("episodes_detail", []):
                if ep.get("flag_get"):
                    found[row["stage"]].append({"x_pos": ep["x_pos"], "seed": ep["seed"],
                                                "checkpoint": data.get("policy"), "source": path})
    return found


def entry(stage: str, records: list[dict]) -> dict:
    if not records:
        return {"flag_x": None, "method": "not measured",
                "note": "no recorded flag episode; the agent has never finished this stage"}
    counts = collections.Counter(r["x_pos"] for r in records)
    sources = sorted({r["source"] for r in records})
    if len(counts) > 1:  # disagreement means one of them is not the flagpole; do not average them
        return {"flag_x": None, "method": "inconsistent, not used",
                "note": f"recorded flag episodes disagree: {dict(sorted(counts.items()))}",
                "episodes": len(records), "sources": sources}
    return {
        "flag_x": int(next(iter(counts))),
        "method": "x_pos at flag_get, read from game RAM in recorded eval episodes",
        "episodes": len(records),
        "agreement": "all recorded flag episodes gave the same value",
        "sources": sources,
    }


def verify(stage: str, record: dict) -> dict:
    """Replay one recorded flag episode in the emulator and read the flagpole x_pos again."""
    import numpy as np
    import torch

    from agent.macros import MacroStepper
    from agent.ppo import PPOAgent
    from config import PPOConfig
    from env.tiles import TileMarioEnv

    ckpt_path = record["checkpoint"]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = PPOConfig.from_dict({**ckpt["config"], "noop_max": 30, "practice_prob": 0.0, "procgen_prob": 0.0})
    env = TileMarioEnv(**cfg.env_kwargs([stage]))
    stepper = MacroStepper(env, cfg.actions)
    agent = PPOAgent(cfg, env.n_policy_actions, env.n_extras)
    agent.net.load_state_dict(ckpt["model"])
    agent.net.eval()
    try:
        seed = record["seed"]
        gen = torch.Generator().manual_seed(seed)
        obs, _ = env.reset(seed=seed, stage=stage, practice=False)
        while True:
            a = int(agent.act({k: v[None] for k, v in obs.items()}, generator=gen)[0][0])
            obs, _, terminated, truncated, info = stepper.step(a)
            if terminated or truncated:
                break
        ep = info["episode"]
        return {"checkpoint": ckpt_path, "seed": seed, "flag_get": bool(ep["flag_get"]), "x_pos": int(ep["x_pos"]),
                "matches_recorded": bool(ep["flag_get"]) and int(ep["x_pos"]) == record["x_pos"]}
    finally:
        env.close()


def main():
    from env.tiles import ALL_STAGES

    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="runs/*/eval*.json")
    p.add_argument("--verify", default=None, help="stage name, or 'all', to replay in the emulator")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    found = harvest(args.pattern)
    stages = {s: entry(s, found.get(s, [])) for s in ALL_STAGES}

    if args.verify:
        targets = [s for s in ALL_STAGES if found.get(s)] if args.verify == "all" else [args.verify]
        for stage in targets:
            if not found.get(stage):
                print(f"{stage}: no recorded flag episode to replay")
                continue
            result = verify(stage, found[stage][0])
            stages[stage]["verified_in_emulator"] = result
            ok = "OK" if result["matches_recorded"] else "MISMATCH"
            print(f"{stage}: replayed seed {result['seed']} -> x_pos {result['x_pos']} "
                  f"(recorded {found[stage][0]['x_pos']}) {ok}", flush=True)

    known = {s: e["flag_x"] for s, e in stages.items() if e["flag_x"] is not None}
    args.out.write_text(json.dumps(
        {"_about": ABOUT, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stages": stages}, indent=2))
    for s in ALL_STAGES:
        flag_x = stages[s]["flag_x"]
        print(f"{s}: {flag_x if flag_x is not None else '-':>6}  {stages[s]['method']}")
    print(f"\n{len(known)} of {len(ALL_STAGES)} stages measured -> {args.out}")


if __name__ == "__main__":
    main()
