"""Training-curve summary for a PPO run: python -m scripts.summarize_ppo runs/ppo_1h_a [window]"""
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

run = Path(sys.argv[1])
window = int(sys.argv[2]) if len(sys.argv) > 2 else 50
updates = list(csv.DictReader(open(run / "updates.csv")))
episodes = list(csv.DictReader(open(run / "episodes.csv")))


def mean(rows, key):
    vals = [float(r[key]) for r in rows if r[key] not in ("", None)]
    return statistics.mean(vals) if vals else float("nan")


print(f"{'updates':>11} {'step':>9} {'x_pos':>6} {'flag':>6} {'coins':>5} {'stall':>5} {'entropy':>7} {'expl_var':>8} {'sps':>5}")
for lo in range(0, len(updates), window):
    g = updates[lo:lo + window]
    print(f"{lo:>5}-{lo + len(g):<5} {g[-1]['step']:>9} {mean(g, 'mean_x_pos'):6.0f} {mean(g, 'flag_rate'):6.3f} "
          f"{mean(g, 'mean_coins'):5.2f} {mean(g, 'stall_rate'):5.2f} {mean(g, 'entropy'):7.3f} "
          f"{mean(g, 'explained_variance'):8.3f} {mean(g, 'steps_per_sec'):5.0f}")

flags = Counter(e["stage"] for e in episodes if e["flag_get"] == "True")
print(f"\nepisodes={len(episodes)} flags={sum(flags.values())} by stage={dict(flags)}")
per_stage = {}
for e in episodes[-2000:]:
    per_stage.setdefault(e["stage"], []).append(float(e["x_pos"]))
print("recent mean x_pos by train stage:",
      {s: round(statistics.mean(v)) for s, v in sorted(per_stage.items())})
