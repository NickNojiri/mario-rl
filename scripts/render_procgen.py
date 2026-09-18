"""Visual check of generated terrain: game screen (original graphics) vs the tile grid (what Mario collides with).

    python -m scripts.render_procgen --stage 1-1 --difficulty 1.0 --seed 3 --steps 40,80,120

Mario is driven by a simple scripted runner (run right, jump when the tile hint says a pit or wall is close),
so the images show whether he stands on, bumps into and falls into the generated terrain.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from env.tiles import COLS, ENEMY_BASE, MARIO_ID, ROWS, TileMarioEnv

p = argparse.ArgumentParser()
p.add_argument("--stage", default="1-1")
p.add_argument("--difficulty", type=float, default=1.0)
p.add_argument("--seed", type=int, default=3)
p.add_argument("--steps", default="40,80,120")
p.add_argument("--out", type=Path, default=Path("runs/procgen_check"))
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

env = TileMarioEnv(stages=[args.stage], noop_max=0, no_progress_steps=10_000, obs_version=4)
obs, info = env.reset(seed=args.seed, stage=args.stage, procgen=True, difficulty=args.difficulty)
RIGHT_B, RIGHT_AB = 3, 4
step, hold = 0, 0


def draw(grid, screen, path):
    s = 3
    left = cv2.resize(cv2.cvtColor(screen, cv2.COLOR_RGB2BGR), (256 * s, 240 * s), interpolation=cv2.INTER_NEAREST)
    right = np.full_like(left, 255)
    for r in range(ROWS):
        for c in range(COLS):
            v = int(grid[r, c])
            x0, y0 = c * 16 * s, (32 + r * 16) * s
            col = ((220, 90, 30) if v == MARIO_ID else (40, 40, 220) if v >= ENEMY_BASE else
                   (40, 160, 214) if v in (0xC0, 0xC1) else (120, 120, 120) if v else None)
            if col:
                cv2.rectangle(right, (x0, y0), (x0 + 16 * s - 1, y0 + 16 * s - 1), col, -1)
            cv2.rectangle(right, (x0, y0), (x0 + 16 * s - 1, y0 + 16 * s - 1), (225, 225, 225), 1)
    cv2.putText(left, "screen: ORIGINAL graphics", (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(right, "tile grid: GENERATED terrain (what Mario collides with)", (10, 700),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(str(path), np.hstack([left, right]))


for target in sorted(int(s) for s in args.steps.split(",")):
    while step < target:
        hints = obs["extras"][-9:]
        pit_close = hints[0] < 0.25 and hints[1] > 0
        wall_close = hints[2] > 0.2
        if hold == 0 and (pit_close or wall_close) and env.ram[0x1D] == 0:
            hold = 8
        action = RIGHT_AB if hold > 0 else RIGHT_B
        hold = max(0, hold - 1)
        obs, _, terminated, truncated, info = env.step(action)
        step += 1
        if terminated or truncated:
            print(f"episode ended at step {step}: {info['episode']['death_cause']} x={info['x_pos']}")
            break
    path = args.out / f"{args.stage}_d{args.difficulty}_s{args.seed}_step{step:04d}.png"
    draw(obs["tiles"][-1], env.screen, path)
    print(f"{path}  x={info.get('x_pos')}  patched columns={env._patcher.columns_patched} "
          f"flag_col={env._patcher.flag_col}")
    if terminated or truncated:
        break
env.close()
