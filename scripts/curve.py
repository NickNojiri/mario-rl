"""Training-health curve for any PPO run in 10 windows: python -m scripts.curve runs/gen_1h"""
import csv
import statistics
import sys
from pathlib import Path

rows = list(csv.DictReader(open(Path(sys.argv[1]) / "updates.csv")))
KEYS = [("gen_flag_rate", "gen_flag", "{:8.3f}"), ("real_x_pos", "real_x", "{:7.0f}"), ("mean_x_pos", "mean_x", "{:7.0f}"),
        ("entropy", "entropy", "{:8.3f}"), ("explained_variance", "expl_var", "{:8.3f}"),
        ("value_loss", "v_loss", "{:7.2f}"), ("approx_kl", "kl", "{:6.3f}"), ("left_share", "LEFT", "{:6.3f}"),
        ("pit_death_rate", "pit", "{:5.2f}")]


def mean(g, k):
    v = [float(x[k]) for x in g if x.get(k) not in ("", None)]
    return statistics.mean(v) if v else float("nan")


w = max(1, len(rows) // 10)
print(f"{'updates':>11} {'step':>9} " + " ".join(f"{h:>{len(fmt.format(0))}}" for _, h, fmt in KEYS))
for i in range(0, len(rows), w):
    g = rows[i:i + w]
    print(f"{i:>5}-{i + len(g):<5} {g[-1]['step']:>9} " + " ".join(fmt.format(mean(g, k)) for k, _, fmt in KEYS))
