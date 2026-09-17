"""Behaviour over training time for a run: deaths, stomps, jumps. python -m scripts.behaviour_trend runs/v3b_1h [n_windows]"""
import csv
import sys
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
windows = int(sys.argv[2]) if len(sys.argv) > 2 else 5
eps = list(csv.DictReader(open(run / "episodes.csv")))
if not eps:
    sys.exit("no episodes yet")
size = max(1, len(eps) // windows)
print(f"{run.name}: {len(eps)} episodes, last logged step {eps[-1]['step']}")
print(f"{'episodes':>13} {'step':>9} {'x_pos':>6} {'pit':>5} {'enemy':>6} {'stall':>6} {'flag':>5} "
      f"{'stomps/ep':>9} {'pts/ep':>7} {'jumps/ep':>8} {'LEFT':>5}")
for w in range(0, len(eps), size):
    g = eps[w:w + size]
    n = len(g)
    c = Counter(e["death_cause"] for e in g)
    steps = sum(int(e["length"]) for e in g) or 1
    print(f"{w:>6}-{w + n:<6} {g[-1]['step']:>9} {sum(int(e['x_pos']) for e in g) / n:6.0f} "
          f"{c['pit'] / n:5.0%} {sum(v for k, v in c.items() if k.startswith('enemy')) / n:6.0%} "
          f"{c['stall'] / n:6.0%} {c['flag'] / n:5.1%} {sum(int(e['point_events']) for e in g) / n:9.2f} "
          f"{sum(int(e['points']) for e in g) / n:7.0f} {sum(int(e['jumps']) for e in g) / n:8.1f} "
          f"{sum(int(e['left_presses']) for e in g) / steps:5.1%}")
killers = Counter(e["death_cause"] for e in eps[-size:] if e["death_cause"].startswith("enemy"))
print("recent enemy killers:", dict(killers.most_common(6)))
