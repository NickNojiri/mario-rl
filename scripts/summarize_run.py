"""Summarize episodes.csv in step buckets: python scripts/summarize_run.py runs/dqn_1_1 [bucket_steps]"""
import csv
import sys
from pathlib import Path

run = Path(sys.argv[1])
bucket = int(sys.argv[2]) if len(sys.argv) > 2 else 250_000
rows = list(csv.DictReader(open(run / "episodes.csv")))

groups: dict[int, list[dict]] = {}
for r in rows:
    groups.setdefault(int(r["step"]) // bucket, []).append(r)

print(f"{'steps':>17} {'eps':>5} {'episodes':>8} {'reward':>7} {'x_pos':>6} {'x>722':>6} {'flags':>5} {'trunc%':>6} {'mean_q':>7} {'sps':>5}")
for k in sorted(groups):
    g = groups[k]
    n = len(g)
    mean = lambda f: sum(float(r[f]) for r in g if r[f] != "") / max(1, sum(1 for r in g if r[f] != ""))
    past_pipe = sum(int(r["x_pos"]) > 722 for r in g) / n
    flags = sum(int(r["flag_get"]) for r in g)
    print(f"{k*bucket:>8,}-{(k+1)*bucket:<8,} {mean('epsilon'):>5.2f} {n:>8} {mean('reward'):>7.0f} {mean('x_pos'):>6.0f} "
          f"{past_pipe:>6.0%} {flags:>5} {mean('truncated'):>6.1%} {mean('mean_q'):>7.1f} {mean('steps_per_sec'):>5.0f}")

total_flags = sum(int(r["flag_get"]) for r in rows)
best = max(rows, key=lambda r: int(r["x_pos"]))
print(f"\nepisodes={len(rows)} total_flags={total_flags} best_x_pos={best['x_pos']} (episode {best['episode']}, step {best['step']})")
