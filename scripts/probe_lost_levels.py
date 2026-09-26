"""Does our RAM map hold on Lost Levels? Answer before treating it as a final test set.

    python -m scripts.probe_lost_levels

The held-out five stages have been used for model selection, so they are really a validation set and our
headline numbers are optimistically biased. There is no spare real level: 32 total = 22 train + 5 held out +
5 excluded. Lost Levels (the Japanese Super Mario Bros 2) ships with gym_super_mario_bros as
`SuperMarioBros2-<world>-<stage>-v0` and is 32 more human-designed levels we have never touched, on what
should be the same engine.

"Should be" is the part that needs checking. Lost Levels has no registered per-stage gym id, only
`SuperMarioBros2-v0` for the whole game, so individual stages are reached by constructing SuperMarioBrosEnv
with lost_levels=True and an explicit target. That is the old gym API: step returns a 4-tuple.

This probe verifies, per stage, that the env loads, that our x_pos addresses (0x6D page + 0x86 offset) agree
with the game's own reported x_pos, that the metatile buffer decodes to something with ground in it, and that
Mario actually moves right. It changes nothing; it only reports whether the observation pipeline would be
valid there.
"""
from __future__ import annotations

import warnings

import numpy as np
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace

from env.tiles import COLS, ROWS, SIMPLE_MOVEMENT

warnings.filterwarnings("ignore")

RIGHT_B = SIMPLE_MOVEMENT.index(["right", "B"])
# gym_super_mario_bros refuses lost-levels worlds 5-12 ("not supported"), so the usable set is worlds 1-4:
# 16 stages, not 32. Probed and confirmed; do not widen this range without re-probing.
LOST_LEVELS_WORLDS = range(1, 5)
STAGES = [(w, s) for w in LOST_LEVELS_WORLDS for s in range(1, 5)]


def probe(world: int, stage: int, steps: int = 60) -> dict:
    out: dict = {"stage": f"{world}-{stage}", "loads": False, "x_pos_matches": None, "has_ground": None,
                 "moves_right": None, "error": None}
    env = base = None
    try:
        base = SuperMarioBrosEnv(lost_levels=True, target=(world, stage))
        env = JoypadSpace(base, SIMPLE_MOVEMENT)
        env.reset()
        _, _, _, info = env.step(RIGHT_B)  # old gym API: no truncated
        out["loads"] = True

        ram = base.ram
        our_x = int(ram[0x6D]) * 256 + int(ram[0x86])
        out["x_pos_matches"] = our_x == int(info["x_pos"])
        out["x_start"] = our_x

        page = np.array(ram[0x500:0x6A0], dtype=np.int16).reshape(2, ROWS, COLS)
        out["has_ground"] = bool((page != 0).any())
        out["solid_tiles"] = int((page != 0).sum())

        for _ in range(steps):
            _, _, done, info = env.step(RIGHT_B)
            if done:
                break
        out["x_end"] = int(info["x_pos"])
        out["moves_right"] = out["x_end"] > out["x_start"]
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()
        elif base is not None:
            base.close()
    return out


if __name__ == "__main__":
    rows = [probe(w, s) for w, s in STAGES]
    ok = 0
    for r in rows:
        if r["error"]:
            print(f"{r['stage']}: FAILED  {r['error']}")
            continue
        good = r["x_pos_matches"] and r["has_ground"] and r["moves_right"]
        ok += bool(good)
        print(f"{r['stage']}: {'OK    ' if good else 'SUSPECT'} "
              f"x {r['x_start']}->{r['x_end']}  x_addr_match={r['x_pos_matches']}  solid={r['solid_tiles']}")
    print(f"\n{ok} of {len(rows)} Lost Levels stages pass the RAM-map probe")
    print("A pass means the observation pipeline is valid there, not that the levels are suitable as a test set.")
