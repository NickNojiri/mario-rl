"""When does SMB write new level columns into the metatile buffer, and how far ahead of the screen?

    python -m scripts.probe_columns --stage 1-1 --steps 80

Tracks the area parser's render position (candidate RAM 0x0725 page / 0x0726 column) and diffs the
0x500-0x69F buffer every agent step to see which level columns were (re)written.
"""
import argparse

import numpy as np

from env.tiles import TileMarioEnv, screen_left_x

p = argparse.ArgumentParser()
p.add_argument("--stage", default="1-1")
p.add_argument("--steps", type=int, default=80)
args = p.parse_args()

env = TileMarioEnv(stages=[args.stage], noop_max=0, no_progress_steps=10_000)
env.reset(seed=0, stage=args.stage)
ram = env.ram
prev = np.array(ram[0x500:0x6A0], dtype=np.int16)
for step in range(args.steps):
    _, _, term, trunc, _ = env.step(3)  # right + B
    cur = np.array(ram[0x500:0x6A0], dtype=np.int16)
    changed = np.flatnonzero(cur != prev)
    prev = cur
    left = screen_left_x(ram)
    render_col = int(ram[0x0725]) * 16 + int(ram[0x0726])
    slots = sorted({(int(i) // 208, (int(i) % 208) % 16) for i in changed})  # (page, column) slots touched
    if step % 4 == 0 or slots:
        print(f"step {step:3d} screen_left_col={left // 16:4d} right_edge_col={(left + 255) // 16:4d} "
              f"render_col(0x725*16+0x726)={render_col:4d} buffer slots written={slots}")
    if term or trunc:
        print("ended", step)
        break
env.close()
