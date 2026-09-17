"""Compare two eval_stages.py JSONs (e.g. policy vs random) per stage with standard errors.

    python -m scripts.compare_evals runs/ppo_1h_a/eval_final_all.json runs/ppo_1h_a/eval_random_all.json
"""
import json
import math
import sys


def load(path):
    d = json.load(open(path))
    return {r["stage"]: r for r in d["stages"]}, d


a_rows, a = load(sys.argv[1])
b_rows, b = load(sys.argv[2])
print(f"A = {a['policy']}\nB = {b['policy']}\n")
print(f"{'group':>5} {'stage':>5} {'A x_pos':>12} {'B x_pos':>12} {'diff':>7} {'z':>5}  {'A flags':>7}")
groups = {}
for stage, ra in a_rows.items():
    rb = b_rows.get(stage)
    if rb is None:
        continue
    se_a = ra["std_x_pos"] / math.sqrt(ra["episodes"])
    se_b = rb["std_x_pos"] / math.sqrt(rb["episodes"])
    diff = ra["mean_x_pos"] - rb["mean_x_pos"]
    se = math.sqrt(se_a ** 2 + se_b ** 2)
    z = diff / se if se > 0 else float("inf")
    groups.setdefault(ra["group"], []).append((diff, se, ra, rb))
    print(f"{ra['group']:>5} {stage:>5} {ra['mean_x_pos']:7.0f}+-{se_a:<4.0f} {rb['mean_x_pos']:7.0f}+-{se_b:<4.0f} "
          f"{diff:+7.0f} {z:5.1f}  {ra['flag_rate']:7.2f}")

print()
for g, items in groups.items():
    n = len(items)
    mean_a = sum(i[2]["mean_x_pos"] for i in items) / n
    mean_b = sum(i[3]["mean_x_pos"] for i in items) / n
    se = math.sqrt(sum(i[1] ** 2 for i in items)) / n
    ahead = sum(1 for d, s, *_ in items if d > 2 * s)
    behind = sum(1 for d, s, *_ in items if d < -2 * s)
    print(f"{g}: A {mean_a:.0f} vs B {mean_b:.0f}  ratio {mean_a / mean_b:.2f}x  diff {mean_a - mean_b:+.0f} "
          f"(SE {se:.0f}, z {(mean_a - mean_b) / se:.1f})  stages clearly ahead {ahead}/{n}, clearly behind {behind}/{n}")
