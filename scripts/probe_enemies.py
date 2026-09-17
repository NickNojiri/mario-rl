"""Verify enemy RAM tables against the emulator: positions must move in the direction the speed says.

    python -m scripts.probe_enemies --stage 1-1 --steps 60

Tables (SMB SprObject layout: slot 0 = player at the base address, enemy i at base + 1 + i):
  flag 0x0F+i, id 0x16+i, state 0x1E+i, x page 0x6E+i, x 0x87+i, y 0xCF+i, x speed 0x58+i, y speed 0xA0+i,
  vertical screen page 0xB6+i. Player: x speed 0x57, y speed 0x9F (both already used in env/tiles.py).
"""
import argparse

from env.tiles import TileMarioEnv, screen_left_x

p = argparse.ArgumentParser()
p.add_argument("--stage", default="1-1")
p.add_argument("--steps", type=int, default=60)
p.add_argument("--action", type=int, default=0, help="SIMPLE_MOVEMENT index held every step (0 = NOOP)")
args = p.parse_args()

env = TileMarioEnv(stages=[args.stage], noop_max=0, no_progress_steps=10_000)
env.reset(seed=0, stage=args.stage)


def s8(v):
    v = int(v)
    return v - 256 if v > 127 else v


prev = {}
for step in range(args.steps):
    _, _, term, trunc, _ = env.step(args.action if step > 5 else 1)
    ram = env.ram
    left = screen_left_x(ram)
    rows = []
    for i in range(6):
        if ram[0x0F + i] == 0:
            continue
        x = int(ram[0x6E + i]) * 256 + int(ram[0x87 + i])
        dx = x - prev.get(i, x)
        prev[i] = x
        rows.append(f"slot{i} id=0x{int(ram[0x16 + i]):02x} state=0x{int(ram[0x1E + i]):02x} sx={x - left:4d} "
                    f"y={int(ram[0xCF + i]):3d} vx={s8(ram[0x58 + i]):4d} vy={s8(ram[0xA0 + i]):4d} "
                    f"page_y={int(ram[0xB6 + i])} moved_dx={dx:+d}")
    if rows and step % 2 == 0:
        print(f"step {step:3d} mario_vx={s8(ram[0x57]):4d} | " + " || ".join(rows))
    if term or trunc:
        print("episode ended at", step)
        break
env.close()
