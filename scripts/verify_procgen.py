"""Does Mario physically interact with generated terrain? Scripted runner over many seeds.

    python -m scripts.verify_procgen --stage 1-1 --seeds 20

Evidence of real collision with generated terrain:
  - pit deaths at x positions where the original stage has solid ground (1-1: first real pit at x ~1100)
  - Mario standing (on-ground state) above the normal floor, i.e. on generated raised ground
Also reports how many runs reach the flag (end of level) and flag handling.
"""
import argparse
from collections import Counter

from env.tiles import TileMarioEnv

p = argparse.ArgumentParser()
p.add_argument("--stage", default="1-1")
p.add_argument("--seeds", type=int, default=20)
p.add_argument("--difficulty", type=float, default=0.5)
p.add_argument("--no-enemies", action="store_true", help="verification only: clear enemies to test terrain alone")
args = p.parse_args()

RIGHT_B, RIGHT_AB = 3, 4
env = TileMarioEnv(stages=[args.stage], noop_max=0, no_progress_steps=150, obs_version=4)
causes, raised, max_xs, flags, early_pits = Counter(), 0, [], 0, 0
for seed in range(args.seeds):
    obs, _ = env.reset(seed=seed, stage=args.stage, procgen=True, difficulty=args.difficulty)
    ram = env.ram
    hold, stood_high = 0, False
    for _ in range(3000):
        hints = obs["extras"][-9:]
        danger = (hints[0] < 0.2 and hints[1] > 0) or hints[2] > 0.2
        if hold == 0 and danger and ram[0x1D] == 0:
            hold = 8
        obs, _, terminated, truncated, info = env.step(RIGHT_AB if hold else RIGHT_B)
        hold = max(0, hold - 1)
        if args.no_enemies:
            for i in range(6):
                ram[0x0F + i] = 0
        if ram[0x1D] == 0 and ram[0xB5] == 1 and int(ram[0xCE]) < 170:  # on ground above the normal floor
            stood_high = True
        if terminated or truncated:
            break
    ep = info["episode"]
    causes[ep["death_cause"]] += 1
    raised += stood_high
    max_xs.append(ep["max_x"])
    flags += ep["flag_get"]
    early_pits += ep["death_cause"] == "pit" and ep["x_pos"] < 1000
env.close()
print(f"stage {args.stage} difficulty {args.difficulty}: {args.seeds} generated levels")
print("episode endings:", dict(causes))
print(f"pit deaths before x=1000 (1-1 has solid ground there): {early_pits}")
print(f"runs where Mario stood on generated raised ground: {raised}/{args.seeds}")
print(f"flags reached: {flags}; mean farthest x {sum(max_xs) / len(max_xs):.0f}, max {max(max_xs)}")
