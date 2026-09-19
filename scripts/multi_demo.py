"""Watch several runs at once in the browser, at 1x-8x speed.

    python -m scripts.multi_demo --checkpoint runs/gen2/latest.pt --envs 4 --speed 2
    python -m scripts.multi_demo --envs 9 --speed 8 --procgen 0.5 --follow      # half generated levels
    then open http://localhost:8766

Real levels are drawn from the game screen. Generated levels are drawn from the tile grid, because the
screen still shows the original level's graphics there. One shared network runs all games in one batch.
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

from agent.macros import MacroStepper
from agent.ppo import PPOAgent
from config import PPOConfig
from env.tiles import COLS, ENEMY_BASE, MARIO_ID, ROWS, TileMarioEnv

CELL_W, CELL_H = 256, 240
BG, INK, MUTED = (27, 24, 24), (240, 240, 240), (158, 150, 150)
C_SOLID, C_BLOCK, C_ENEMY, C_MARIO, C_GOOD, C_BAD = ((140, 130, 130), (40, 160, 214), (70, 70, 220),
                                                     (235, 140, 70), (100, 190, 60), (90, 90, 230))

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, default=Path("runs/gen2/latest.pt"))
p.add_argument("--follow", action="store_true", help="reload the checkpoint whenever training saves it")
p.add_argument("--envs", type=int, default=4)
p.add_argument("--speed", type=float, default=2.0, help="1x = real NES speed; 8x = eight times faster")
p.add_argument("--fps", type=int, default=20, help="stream frame rate")
p.add_argument("--scale", type=int, default=2, help="pixel scale per cell")
p.add_argument("--stages", default="test", help="test | train | comma list")
p.add_argument("--procgen", type=float, default=0.0, help="fraction of cells playing generated levels")
p.add_argument("--difficulty", type=float, default=1.0)
p.add_argument("--port", type=int, default=8766)
p.add_argument("--greedy", action="store_true")
args = p.parse_args()

torch.set_num_threads(2)
ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
cfg = PPOConfig.from_dict({**ckpt["config"], "noop_max": 30, "practice_prob": 0.0, "procgen_prob": 0.0})
train_stages, test_stages = cfg.resolved_stages()
stages = {"test": test_stages, "train": train_stages}.get(args.stages) or args.stages.split(",")

envs = [TileMarioEnv(**cfg.env_kwargs(stages)) for _ in range(args.envs)]
steppers = [MacroStepper(e, cfg.actions) for e in envs]
agent = PPOAgent(cfg, envs[0].n_policy_actions, envs[0].n_extras)
agent.net.load_state_dict(ckpt["model"])
agent.net.eval()
loaded = {"mtime": args.checkpoint.stat().st_mtime, "step": ckpt.get("global_step", 0)}
rng = np.random.default_rng(0)
cells = [{"queue": [], "banner": None, "info": {}, "episodes": 0, "gen": False} for _ in envs]

GRID_COLS = int(np.ceil(np.sqrt(args.envs)))
GRID_ROWS = int(np.ceil(args.envs / GRID_COLS))
W, H = CELL_W * args.scale, CELL_H * args.scale
PAGE = f"""<!doctype html><html><head><meta charset="utf-8"><title>mario-rl multi view</title>
<style>html,body{{margin:0;height:100%;background:#18181b;color:#e4e4e7;font-family:system-ui,sans-serif}}
main{{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px}}
img{{max-width:98vw;max-height:90vh;image-rendering:pixelated;border-radius:6px}}
p{{margin:0;color:#a1a1aa;font-size:14px}}</style></head><body><main>
<img src="/stream" alt="live agent stream">
<p>{args.envs} games at {args.speed:g}x &middot; generated levels drawn from the tile grid &middot;
github.com/NickNojiri/mario-rl</p></main></body></html>""".encode()

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
            self.end_headers()
            seen = -1
            try:
                while True:
                    with cond:
                        cond.wait_for(lambda: latest["id"] != seen, timeout=5)
                        jpeg, seen = latest["jpeg"], latest["id"]
                    if jpeg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
server.daemon_threads = True
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"multi view streaming at http://localhost:{args.port}", flush=True)


def text(img, s, xy, color=INK, size=0.45, bold=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, bold, cv2.LINE_AA)


def grid_image(env) -> np.ndarray:
    """Draw the collision grid (used for generated levels, which the screen can't show)."""
    img = np.full((CELL_H, CELL_W, 3), BG, np.uint8)
    grid = env._obs()["tiles"][-1]
    for r in range(ROWS):
        for c in range(COLS):
            v = int(grid[r, c])
            col = (C_MARIO if v == MARIO_ID else C_ENEMY if v >= ENEMY_BASE else
                   C_BLOCK if v in (0xC0, 0xC1, 0xC2) else C_SOLID if v else None)
            if col:
                cv2.rectangle(img, (c * 16, 32 + r * 16), (c * 16 + 15, 32 + r * 16 + 15), col, -1)
    return img


def cell_image(env, cell) -> np.ndarray:
    img = grid_image(env) if cell["gen"] else cv2.cvtColor(env.screen, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    label = f"{env.stage}{' GEN d' + format(cell['difficulty'], '.1f') if cell['gen'] else ''}"
    cv2.rectangle(img, (0, 0), (W, 22 * args.scale), (0, 0, 0), -1)
    text(img, label, (6, 15 * args.scale), C_GOOD if cell["gen"] else INK, 0.4 * args.scale, 1)
    text(img, f"x {cell['info'].get('x_pos', 0)}", (W - 60 * args.scale, 15 * args.scale), MUTED, 0.4 * args.scale)
    if cell["banner"]:
        msg, color = cell["banner"]
        cv2.rectangle(img, (0, H // 2 - 16 * args.scale), (W, H // 2 + 8 * args.scale), (0, 0, 0), -1)
        text(img, msg, (8, H // 2), color, 0.42 * args.scale, 2)
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (60, 60, 60), 1)
    return img


def reset_cell(i):
    env, cell = envs[i], cells[i]
    gen = rng.random() < args.procgen
    stage = stages[(cell["episodes"] + i) % len(stages)] if not gen else None
    obs, info = env.reset(seed=int(rng.integers(1 << 30)), stage=stage, procgen=gen, practice=False,
                          difficulty=args.difficulty if gen else None)
    cell.update(obs=obs, queue=[], banner=None, info={}, gen=bool(info.get("procgen")),
                difficulty=args.difficulty if gen else 0.0)
    cell["episodes"] += 1


def maybe_reload():
    if not args.follow:
        return
    try:
        mtime = args.checkpoint.stat().st_mtime
        if mtime != loaded["mtime"]:
            new = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            agent.net.load_state_dict(new["model"])
            loaded.update(mtime=mtime, step=new.get("global_step", 0))
            print(f"reloaded checkpoint at step {loaded['step']}", flush=True)
    except Exception as exc:
        print(f"reload skipped: {exc}", flush=True)


for i in range(len(envs)):
    reset_cell(i)
frame_dt = 1 / (60 * args.speed)
stream_dt = 1 / args.fps
clock = {"frame": time.perf_counter(), "stream": 0.0}
try:
    while True:
        need = [i for i, c in enumerate(cells) if not c["queue"] and not c.get("pending_reset")]
        if need:
            batch = {k: np.stack([cells[i]["obs"][k] for i in need]) for k in cells[need[0]]["obs"]}
            actions, _, _ = agent.act(batch, greedy=args.greedy)
            for i, a in zip(need, actions):
                joy, length = steppers[i].expand(int(a))
                cells[i]["queue"] = [joy] * length
        for i, (env, cell) in enumerate(zip(envs, cells)):
            if not cell["queue"]:  # finished episode, waiting on its banner before restarting
                continue
            obs, _, terminated, truncated, info = env.step(cell["queue"].pop(0))
            cell["obs"], cell["info"] = obs, info
            if terminated or truncated:
                ep = info["episode"]
                cause = ep.get("death_cause", "")
                cell["banner"] = (("LEVEL CLEAR!", C_GOOD) if ep["flag_get"] else
                                  (f"{cause.replace('enemy_', 'enemy ')} x={ep['x_pos']}", C_BAD))
                cell["queue"] = []
                cell["pending_reset"] = time.perf_counter() + 1.2 / args.speed
        for i, cell in enumerate(cells):
            if cell.get("pending_reset") and time.perf_counter() >= cell["pending_reset"]:
                cell["pending_reset"] = None
                maybe_reload()
                reset_cell(i)

        now = time.perf_counter()
        wait = frame_dt - (now - clock["frame"])
        if wait > 0:
            time.sleep(wait)
        clock["frame"] = time.perf_counter()
        if clock["frame"] - clock["stream"] >= stream_dt:
            clock["stream"] = clock["frame"]
            tiles = [cell_image(e, c) for e, c in zip(envs, cells)]
            while len(tiles) < GRID_ROWS * GRID_COLS:
                tiles.append(np.full_like(tiles[0], BG))
            mosaic = np.vstack([np.hstack(tiles[r * GRID_COLS:(r + 1) * GRID_COLS]) for r in range(GRID_ROWS)])
            banner = np.full((28, mosaic.shape[1], 3), BG, np.uint8)
            text(banner, f"{args.checkpoint}  step {loaded['step'] / 1e6:.2f}M   {args.speed:g}x speed   "
                         f"{args.envs} games", (10, 19), MUTED, 0.5)
            ok, buf = cv2.imencode(".jpg", np.vstack([banner, mosaic]), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with cond:
                    latest["jpeg"], latest["id"] = buf.tobytes(), latest["id"] + 1
                    cond.notify_all()
except KeyboardInterrupt:
    pass
finally:
    server.shutdown()
    for e in envs:
        e.close()
