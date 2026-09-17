"""Live demo in the browser: the trained agent playing, next to the tile grid it sees and its button choices.

    python -m scripts.demo                                   # held-out stages, loops forever
    python -m scripts.demo --stages 1-1,3-3 --speed 1.5
    then open http://localhost:8765 (works from Windows; WSL forwards localhost)

Streams MJPEG over a tiny local HTTP server, so no GUI libraries are needed inside WSL.
Ctrl+C stops it.
"""
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch

from agent.ppo import PPOAgent
from config import PPOConfig
from env.tiles import COLS, ENEMY_BASE, MARIO_ID, ROWS, TileMarioEnv, n_extras

SCALE = 3
GAME_W, GAME_H = 256 * SCALE, 240 * SCALE
PANEL_W = 520
CELL = 30
# BGR colors (frames are encoded with OpenCV)
BG, INK, MUTED, GRID = (27, 24, 24), (240, 240, 240), (158, 150, 150), (54, 48, 48)
C_SOLID, C_BLOCK, C_ENEMY, C_MARIO = (140, 130, 130), (40, 160, 214), (70, 70, 220), (235, 140, 70)
C_GOOD, C_BAD = (100, 190, 60), (90, 90, 230)
BUTTONS = ["NOOP", "RIGHT", "RIGHT+A", "RIGHT+B", "RIGHT+A+B", "A", "LEFT"]

PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>mario-rl live demo</title>
<style>html,body{margin:0;height:100%;background:#18181b;color:#e4e4e7;font-family:system-ui,sans-serif}
main{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px}
img{max-width:98vw;max-height:88vh;image-rendering:pixelated;border-radius:6px}
p{margin:0;color:#a1a1aa;font-size:14px}</style></head><body><main>
<img src="/stream" alt="live agent stream">
<p>PPO agent (1-hour run) &middot; tile-grid observation &middot; github.com/NickNojiri/mario-rl</p>
</main></body></html>"""

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, default=Path("runs/ppo_1h_a/latest.pt"))
p.add_argument("--stages", default="test", help="test | train | comma list")
p.add_argument("--seed", type=int, default=7)
p.add_argument("--speed", type=float, default=1.0, help="1.0 = real NES speed (60 fps)")
p.add_argument("--fps", type=int, default=30, help="stream frame rate")
p.add_argument("--port", type=int, default=8765)
p.add_argument("--greedy", action="store_true")
p.add_argument("--once", action="store_true", help="play the stage list once instead of looping")
args = p.parse_args()

torch.set_num_threads(2)
ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
cfg = PPOConfig.from_dict({**ckpt["config"], "noop_max": 30})
train_stages, test_stages = cfg.resolved_stages()
stages = {"test": test_stages, "train": train_stages}.get(args.stages) or args.stages.split(",")
env = TileMarioEnv(**cfg.env_kwargs(stages))
agent = PPOAgent(cfg, env.n_actions, n_extras(env.n_actions))
agent.net.load_state_dict(ckpt["model"])
agent.net.eval()
names = BUTTONS if env.n_actions == len(BUTTONS) else [" ".join(b).upper() for b in env.action_set]

# ------------------------------------------------------------------ stream server
latest = {"jpeg": None, "id": 0}
cond = threading.Condition()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with cond:
                        cond.wait_for(lambda: latest["id"] != seen, timeout=5)
                        jpeg, seen = latest["jpeg"], latest["id"]
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
server.daemon_threads = True
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"demo streaming at http://localhost:{args.port}", flush=True)

# ------------------------------------------------------------------ drawing
state = {"probs": np.zeros(env.n_actions), "action": 0, "value": 0.0, "reward": 0.0, "x": 0, "banner": None,
         "grid": np.zeros((ROWS, COLS), np.int16), "results": []}


def text(img, s, xy, color=INK, size=0.6, bold=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, bold, cv2.LINE_AA)


def draw_panel() -> np.ndarray:
    panel = np.full((GAME_H, PANEL_W, 3), BG, np.uint8)
    held_out = env.stage in test_stages
    text(panel, f"WORLD {env.stage}", (20, 40), INK, 0.95, 2)
    tag, tag_col = ("HELD-OUT: never trained on", C_GOOD) if held_out else ("training stage", MUTED)
    text(panel, tag, (200, 40), tag_col, 0.55, 1)

    text(panel, "What the agent sees: 13x16 tile grid from RAM", (20, 74), MUTED, 0.5)
    ox, oy = 20, 86
    grid = state["grid"]
    for r in range(ROWS):
        for c in range(COLS):
            v = int(grid[r, c])
            x0, y0 = ox + c * CELL, oy + r * CELL
            col = (C_MARIO if v == MARIO_ID else C_ENEMY if v >= ENEMY_BASE else
                   C_BLOCK if v in (0xC0, 0xC1, 0xC2, 0xC3) else C_SOLID if v else None)
            if col:
                cv2.rectangle(panel, (x0 + 1, y0 + 1), (x0 + CELL - 2, y0 + CELL - 2), col, -1)
            else:
                cv2.rectangle(panel, (x0, y0), (x0 + CELL - 1, y0 + CELL - 1), GRID, 1)
    ly = oy + ROWS * CELL + 24
    for lx, (lab, col) in zip((20, 110, 205, 365), [("Mario", C_MARIO), ("enemy", C_ENEMY),
                                                     ("coin / ?-block", C_BLOCK), ("solid", C_SOLID)]):
        cv2.rectangle(panel, (lx, ly - 12), (lx + 14, ly + 2), col, -1)
        text(panel, lab, (lx + 20, ly), MUTED, 0.46)

    by = ly + 32
    text(panel, "Button probabilities (the policy's choice is highlighted)", (20, by), MUTED, 0.5)
    for i, name in enumerate(names):
        y = by + 12 + i * 21
        pr = float(state["probs"][i])
        chosen = i == state["action"]
        text(panel, name, (20, y + 14), INK if chosen else MUTED, 0.46, 2 if chosen else 1)
        cv2.rectangle(panel, (135, y + 3), (135 + int(300 * pr), y + 17), C_MARIO if chosen else (100, 90, 90), -1)
        text(panel, f"{pr:4.0%}", (445, y + 15), INK if chosen else MUTED, 0.46)

    sy = by + 12 + len(names) * 21 + 24
    text(panel, f"x_pos {state['x']:5d}    value estimate {state['value']:6.2f}", (20, sy), INK, 0.52)
    res = state["results"]
    if res:
        text(panel, f"this session: {len(res)} episodes, mean x_pos {np.mean([r[1] for r in res]):.0f}, "
                    f"{sum(r[2] for r in res)} flags", (20, sy + 26), MUTED, 0.46)
    return panel


frame_dt = 1 / (60 * args.speed)
stream_dt = 1 / args.fps
clock = {"frame": time.perf_counter(), "stream": 0.0}


def on_frame():
    now = time.perf_counter()
    wait = frame_dt - (now - clock["frame"])
    if wait > 0:
        time.sleep(wait)
    clock["frame"] = time.perf_counter()
    if clock["frame"] - clock["stream"] < stream_dt:
        return
    clock["stream"] = clock["frame"]
    game = cv2.resize(cv2.cvtColor(env.screen, cv2.COLOR_RGB2BGR), (GAME_W, GAME_H), interpolation=cv2.INTER_NEAREST)
    if state["banner"]:
        msg, col = state["banner"]
        cv2.rectangle(game, (0, GAME_H // 2 - 55), (GAME_W, GAME_H // 2 + 35), (0, 0, 0), -1)
        text(game, msg, (40, GAME_H // 2 + 5), col, 1.2, 3)
    ok, buf = cv2.imencode(".jpg", np.hstack([game, draw_panel()]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        with cond:
            latest["jpeg"], latest["id"] = buf.tobytes(), latest["id"] + 1
            cond.notify_all()


env.frame_callback = on_frame
episode = 0
try:
    while True:
        for stage in stages:
            seed = args.seed + episode
            episode += 1
            gen = torch.Generator().manual_seed(seed)
            obs, _ = env.reset(seed=seed, stage=stage)
            state.update(x=0, banner=None, grid=obs["tiles"][-1])
            while True:
                with torch.no_grad():
                    t = {k: torch.as_tensor(v[None]) for k, v in obs.items()}
                    logits, value = agent.net(t["tiles"], t["extras"])
                    probs = torch.softmax(logits, 1)[0]
                a = int(probs.argmax()) if args.greedy else int(torch.multinomial(probs, 1, generator=gen))
                state.update(probs=probs.numpy(), action=a, value=float(value))
                obs, _, terminated, truncated, info = env.step(a)
                state.update(x=int(info["x_pos"]), grid=obs["tiles"][-1])
                if terminated or truncated:
                    break
            ep = info["episode"]
            state["results"].append((stage, ep["x_pos"], ep["flag_get"]))
            state["banner"] = (("LEVEL CLEAR!", C_GOOD) if ep["flag_get"] else
                               (f"stuck: cut off at x={ep['x_pos']}", C_BAD) if truncated else
                               (f"died at x={ep['x_pos']}", C_BAD))
            print(f"{stage}: x_pos={ep['x_pos']} flag={ep['flag_get']} coins={ep['coins']} steps={ep['length']}",
                  flush=True)
            for _ in range(int(90 * args.speed)):  # hold the banner ~1.5 s
                on_frame()
        if args.once:
            break
except KeyboardInterrupt:
    pass
finally:
    server.shutdown()
    env.close()
    res = state["results"]
    if res:
        print(f"episodes={len(res)} mean x_pos={np.mean([r[1] for r in res]):.0f} flags={sum(r[2] for r in res)}")
