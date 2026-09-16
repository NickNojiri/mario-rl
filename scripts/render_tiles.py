"""Save the game screen next to the decoded tile grid, to check the RAM decoding by eye.

    python -m scripts.render_tiles --stage 1-1 --steps 0,40,90 --out runs/tile_check
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from env.tiles import COLS, ENEMY_BASE, MARIO_ID, ROWS, TileMarioEnv

p = argparse.ArgumentParser()
p.add_argument("--stage", default="1-1")
p.add_argument("--steps", default="0,40,90", help="agent steps (running right, jumping periodically)")
p.add_argument("--out", type=Path, default=Path("runs/tile_check"))
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

env = TileMarioEnv(stages=[args.stage], noop_max=0, no_progress_steps=10_000)
obs, _ = env.reset(seed=0, stage=args.stage)
targets = sorted(int(s) for s in args.steps.split(","))
RIGHT, RIGHT_A = 1, 2  # SIMPLE_MOVEMENT indices
step = 0


def render(grid: np.ndarray, screen: np.ndarray, path: Path):
    scale = 3
    left = cv2.resize(cv2.cvtColor(screen, cv2.COLOR_RGB2BGR), (256 * scale, 240 * scale),
                      interpolation=cv2.INTER_NEAREST)
    # Draw the grid as it maps onto the screen: row r covers y = 32 + 16r, col c covers x = 16c (approx.)
    overlay = left.copy()
    right = np.full_like(left, 255)
    for r in range(ROWS):
        for c in range(COLS):
            v = int(grid[r, c])
            x0, y0 = c * 16 * scale, (32 + r * 16) * scale
            x1, y1 = x0 + 16 * scale - 1, y0 + 16 * scale - 1
            if v == 0:
                color = None
            elif v == MARIO_ID:
                color = (220, 90, 30)
            elif v >= ENEMY_BASE:
                color = (40, 40, 220)
            else:
                color = (120, 120, 120)
            if color:
                cv2.rectangle(right, (x0, y0), (x1, y1), color, -1)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
                label = "M" if v == MARIO_ID else (f"e{v - ENEMY_BASE}" if v >= ENEMY_BASE else f"{v:02x}")
                cv2.putText(right, label, (x0 + 4, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(right, (x0, y0), (x1, y1), (225, 225, 225), 1)
    cv2.imwrite(str(path), np.hstack([overlay, right]))


for target in targets:
    while step < target:
        action = RIGHT_A if (step // 6) % 3 == 2 else RIGHT
        obs, _, terminated, truncated, info = env.step(action)
        step += 1
        if terminated or truncated:
            print(f"episode ended at step {step}")
            break
    grid = obs["tiles"][-1]
    ram = env.ram
    path = args.out / f"{args.stage}_step{step:04d}.png"
    render(grid, env.screen, path)
    print(f"{path}  mario_ram_y=0xCE:{ram[0xCE]} 0x3B8:{ram[0x3B8]}  "
          f"screen_x smb_env={(int(ram[0x86]) - int(ram[0x71C])) % 256} ours="
          f"{int(ram[0x6D]) * 256 + int(ram[0x86]) - (int(ram[0x71A]) * 256 + int(ram[0x71C]))}  "
          f"unique tile ids={sorted(set(int(v) for v in grid.flat))}")
env.close()
