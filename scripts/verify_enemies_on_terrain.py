"""After the lift fix, are generated-level enemies on the surface (never inside solid tiles)?

    python -m scripts.verify_enemies_on_terrain --seeds 20
Counts agent steps where a walking enemy's center sits inside a solid tile, and shows one rendered frame.
"""
import argparse

from env.procgen import WALKERS
from env.tiles import TileMarioEnv, read_tile_grid

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=20)
args = p.parse_args()

env = TileMarioEnv(stages=["1-1"], noop_max=0, no_progress_steps=150, obs_version=4)
buried = checks = lifted = 0
for seed in range(args.seeds):
    obs, _ = env.reset(seed=seed, stage="1-1", procgen=True, difficulty=1.5)
    ram = env.ram
    hold = 0
    for _ in range(400):
        hints = obs["extras"][-9:]
        if hold == 0 and ((hints[0] < 0.2 and hints[1] > 0) or hints[2] > 0.2) and ram[0x1D] == 0:
            hold = 8
        obs, _, term, trunc, _ = env.step(4 if hold else 3)
        hold = max(0, hold - 1)
        for i in range(6):
            if ram[0x0F + i] and int(ram[0x16 + i]) in WALKERS and ram[0xB6 + i] == 1:
                x = int(ram[0x6E + i]) * 256 + int(ram[0x87 + i]) + 8
                row = (int(ram[0xCF + i]) + 8 - 32) // 16
                page, col = (x // 16 // 16) % 2, (x // 16) % 16
                if 0 <= row < 13:
                    checks += 1
                    buried += ram[0x500 + page * 208 + row * 16 + col] != 0
        if term or trunc:
            break
    lifted += env._patcher.enemies_lifted
env.close()
print(f"walker-steps checked {checks}, inside solid tiles after the fix: {buried}, lifts performed: {lifted}")
