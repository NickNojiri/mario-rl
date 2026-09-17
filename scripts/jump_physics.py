"""Measure Super Mario Bros jump physics with scripted inputs on an open flat runway.

    python -m scripts.jump_physics --stage 8-1

For each run-up (walk or run, several lengths) and each A-hold duration (in agent steps = 4 frames), records
speed at takeoff, distance travelled until Mario lands back on the ground, peak height and airtime.
This is ground truth for "what jump is needed to clear a gap of N tiles" at the agent's decision resolution.
"""
import argparse
import csv
from pathlib import Path

from env.tiles import TileMarioEnv, _s8

RIGHT, RIGHT_A, RIGHT_B, RIGHT_AB = 1, 2, 3, 4  # SIMPLE_MOVEMENT

p = argparse.ArgumentParser()
p.add_argument("--stage", default="8-1")
p.add_argument("--out", type=Path, default=Path("runs/analysis/jump_physics.csv"))
args = p.parse_args()

env = TileMarioEnv(stages=[args.stage], noop_max=0, no_progress_steps=10_000)
rows = []
for mode, runup_action, jump_action, release_action in (("walk", RIGHT, RIGHT_A, RIGHT), ("run", RIGHT_B, RIGHT_AB, RIGHT_B)):
    for runup in (0, 6, 12, 20):
        for hold in (1, 2, 3, 4, 5, 6, 8, 10, 12):
            env.reset(seed=0, stage=args.stage)
            ram = env.ram
            dead = False
            for _ in range(runup):
                _, _, term, trunc, _ = env.step(runup_action)
                dead |= term or trunc
                if dead:
                    break
            x0 = int(ram[0x6D]) * 256 + int(ram[0x86])
            y0 = int(ram[0xCE])
            vx0 = _s8(ram[0x57])
            peak, air_steps, landed = 0, 0, False
            for t in range(0 if dead else 60):
                _, _, term, trunc, _ = env.step(jump_action if t < hold else release_action)
                dead |= term or trunc
                y = int(ram[0xCE])
                peak = max(peak, y0 - y)
                air_steps += 1
                if t > 0 and int(ram[0x1D]) == 0:  # float state back to "on ground"
                    landed = True
                    break
                if dead:
                    break
            x1 = int(ram[0x6D]) * 256 + int(ram[0x86])
            rows.append({"mode": mode, "runup_steps": runup, "takeoff_speed": vx0, "hold_steps": hold,
                         "hold_frames": hold * 4, "distance_px": x1 - x0, "distance_tiles": round((x1 - x0) / 16, 2),
                         "peak_px": peak, "peak_tiles": round(peak / 16, 2), "air_frames": air_steps * 4,
                         "valid": landed and not dead and int(ram[0xCE]) == y0})
env.close()

args.out.parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print(f"{'mode':>4} {'runup':>5} {'speed':>5} | " + " ".join(f"A{h:>2}".rjust(9) for h in (1, 2, 3, 4, 5, 6, 8, 10, 12)))
for mode in ("walk", "run"):
    for runup in (0, 6, 12, 20):
        rs = [r for r in rows if r["mode"] == mode and r["runup_steps"] == runup]
        cells = [(f"{r['distance_tiles']:.1f}t/{r['peak_tiles']:.1f}h" if r["valid"] else "   (x)   ") for r in rs]
        print(f"{mode:>4} {runup:>5} {rs[0]['takeoff_speed']:>5} | " + " ".join(c.rjust(9) for c in cells))
print("cells: horizontal distance in tiles / peak height in tiles; (x) = hit something or died, not clean")
print(f"wrote {args.out}")
