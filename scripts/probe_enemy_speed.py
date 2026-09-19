"""Can enemies be made faster by scaling their RAM x speed once when they spawn? And can the clock be shortened?

    python -m scripts.probe_enemy_speed
"""
from env.tiles import TileMarioEnv, _s8


def goomba_motion(factor: float, steps: int = 12):
    env = TileMarioEnv(stages=["1-1"], noop_max=0, no_progress_steps=10_000)
    env.reset(seed=0, stage="1-1")
    ram = env.ram
    scaled, xs = set(), []
    for t in range(60):
        env.step(1 if t < 22 else 0)  # walk until the first goomba spawns (~step 20), then stand and watch it
        for i in range(6):
            if ram[0x0F + i] and ram[0x16 + i] == 0x06:
                if i not in scaled and factor != 1.0:
                    v = _s8(ram[0x58 + i])
                    ram[0x58 + i] = max(-127, min(127, int(round(v * factor)))) & 0xFF
                    scaled.add(i)
                xs.append((int(ram[0x6E + i]) * 256 + int(ram[0x87 + i]), _s8(ram[0x58 + i])))
                break
        if len(xs) >= steps:
            break
    env.close()
    moves = [a[0] - b[0] for b, a in zip(xs, xs[1:])]
    return moves, [v for _, v in xs]


for f in (1.0, 2.0, 3.0):
    moves, speeds = goomba_motion(f)
    print(f"factor {f}: goomba px per agent step {moves}  ram speed {sorted(set(speeds))}")

env = TileMarioEnv(stages=["1-1"], noop_max=0, no_progress_steps=10_000)
env.reset(seed=0, stage="1-1")
ram = env.ram
for addr, digit in zip((0x07F8, 0x07F9, 0x07FA), (0, 3, 0)):  # clock = 030
    ram[addr] = digit
print("clock after write:", env._smb._time)
for t in range(2000):
    _, _, term, trunc, info = env.step(0)
    if term or trunc:
        print(f"clock 030: episode ended after {t + 1} steps, time={info['time']}, cause={info['episode']['death_cause']}")
        break
env.close()
