"""Are generated tile themes and pipes solid in the real game? Scripted runner, enemies cleared, easy terrain.

    python -m scripts.verify_themes --seeds 45

If a theme's tile were not solid, Mario would fall through generated ground right after the opening screen
(pit deaths just past x=256) instead of standing on raised ground and reaching the flag.
"""
import argparse
from collections import defaultdict

from env.tiles import TileMarioEnv

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=45)
p.add_argument("--difficulty", type=float, default=0.0)
args = p.parse_args()

RIGHT_B, RIGHT_AB = 3, 4
env = TileMarioEnv(stages=["1-1"], noop_max=0, no_progress_steps=150, obs_version=4)
stats = defaultdict(lambda: {"n": 0, "flag": 0, "stood_high": 0, "fell_early": 0, "pipes_passed": 0, "max_x": 0})
for seed in range(args.seeds):
    obs, _ = env.reset(seed=seed, stage="1-1", procgen=True, difficulty=args.difficulty)
    ram, gen = env.ram, env._patcher.gen
    s = stats[hex(gen.solid)]
    s["n"] += 1
    hold, high = 0, False
    for _ in range(3000):
        hints = obs["extras"][-9:]
        if hold == 0 and ((hints[0] < 0.2 and hints[1] > 0) or hints[2] > 0.2) and ram[0x1D] == 0:
            hold = 8
        obs, _, term, trunc, info = env.step(RIGHT_AB if hold else RIGHT_B)
        hold = max(0, hold - 1)
        for i in range(6):
            ram[0x0F + i] = 0
        high |= ram[0x1D] == 0 and ram[0xB5] == 1 and int(ram[0xCE]) < 170
        if term or trunc:
            break
    ep = info["episode"]
    s["flag"] += ep["flag_get"]
    s["stood_high"] += high
    s["fell_early"] += ep["death_cause"] == "pit" and ep["x_pos"] < 330
    s["max_x"] = max(s["max_x"], ep["max_x"])
env.close()
for theme, s in stats.items():
    print(f"theme {theme}: {s['n']:2d} levels, flags {s['flag']}, stood on raised ground {s['stood_high']}, "
          f"fell through just after the opening screen {s['fell_early']}, farthest x {s['max_x']}")
