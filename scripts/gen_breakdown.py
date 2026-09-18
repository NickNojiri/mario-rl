"""Generated-level run breakdown: learning curve on generated vs real episodes, and completion by difficulty.

    python -m scripts.gen_breakdown runs/gen_1h
"""
import csv
import sys
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
eps = [e for e in csv.DictReader(open(run / "episodes.csv")) if e.get("practice") != "True"]
q = len(eps) // 4
print(f"{run.name}: {len(eps)} episodes ({sum(e['procgen'] == 'True' for e in eps)} generated)")
print(f"{'quarter':>8} {'gen x':>7} {'gen flag':>9} {'real x':>7} {'real flag':>9} {'gen pit':>8} {'gen enemy':>9}")
for k in range(4):
    g = eps[k * q:(k + 1) * q]
    gen = [e for e in g if e["procgen"] == "True"]
    real = [e for e in g if e["procgen"] != "True"]
    c = Counter(e["death_cause"] for e in gen)
    print(f"{k + 1:>8} {sum(int(e['x_pos']) for e in gen) / max(len(gen), 1):7.0f} "
          f"{sum(e['flag_get'] == 'True' for e in gen) / max(len(gen), 1):9.1%} "
          f"{sum(int(e['x_pos']) for e in real) / max(len(real), 1):7.0f} "
          f"{sum(e['flag_get'] == 'True' for e in real) / max(len(real), 1):9.1%} "
          f"{c['pit'] / max(len(gen), 1):8.0%} "
          f"{sum(v for kk, v in c.items() if kk.startswith('enemy')) / max(len(gen), 1):9.0%}")

last = [e for e in eps[len(eps) // 2:] if e["procgen"] == "True" and e["difficulty"] not in ("", None)]
print("\nsecond half, generated levels by difficulty:")
for lo in (0.0, 0.25, 0.5, 0.75):
    b = [e for e in last if lo <= float(e["difficulty"]) < lo + 0.25]
    if b:
        c = Counter(e["death_cause"] for e in b)
        print(f"  difficulty {lo:.2f}-{lo + 0.25:.2f}: {len(b):5d} episodes, flag {c['flag'] / len(b):5.1%}, "
              f"pit deaths {c['pit'] / len(b):4.0%}, enemy deaths "
              f"{sum(v for k, v in c.items() if k.startswith('enemy')) / len(b):4.0%}, "
              f"mean x {sum(int(e['x_pos']) for e in b) / len(b):5.0f}")
