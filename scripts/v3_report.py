"""Side-by-side table of the v3 experiments vs the 1-hour v2 baseline and random.

    python -m scripts.v3_report          # table
    python -m scripts.v3_report --best   # print the run name with the best held-out mean x_pos
"""
import json
import sys
from pathlib import Path

RUNS = [("ppo_1h_a (v2 baseline)", "ppo_1h_a", "runs/ppo_1h_a/eval_final_all_v3diag.json"),
        ("v3b: new reward", "v3b_1h", "runs/v3b_1h/eval_final_all.json"),
        ("v3c: + enemy motion", "v3c_1h", "runs/v3c_1h/eval_final_all.json"),
        ("v3d: + 12 buttons", "v3d_1h", "runs/v3d_1h/eval_final_all.json"),
        ("random (7 buttons)", None, "runs/ppo_1h_a/eval_random_all.json")]
KEYS = [("mean_x_pos", "x_pos", "{:.0f}"), ("flag_rate", "flags", "{:.1%}"), ("pit_death_rate", "pit", "{:.0%}"),
        ("enemy_death_rate", "enemy", "{:.0%}"), ("stall_rate", "stall", "{:.0%}"), ("left_share", "LEFT", "{:.1%}"),
        ("noop_share", "NOOP", "{:.1%}"), ("points_per_episode", "points", "{:.0f}")]

rows = []
for label, run, path in RUNS:
    if Path(path).exists():
        d = json.load(open(path))
        rows.append((label, run, d.get("train_summary", {}), d.get("test_summary", {})))

if "--best" in sys.argv:
    cands = [(r[3].get("mean_x_pos", 0), r[1]) for r in rows if r[1] and r[1].startswith("v3")]
    print(max(cands)[1] if cands else "")
    sys.exit()

for group, idx in (("HELD-OUT stages", 3), ("TRAINING stages", 2)):
    print(f"\n{group}")
    print(f"{'run':26}" + "".join(f"{h:>9}" for _, h, _ in KEYS))
    for r in rows:
        s = r[idx]
        print(f"{r[0]:26}" + "".join(f"{(fmt.format(s[k]) if k in s else '-'):>9}" for k, _, fmt in KEYS))
print("\nNote: the random baseline predates the new diagnostics, so its behaviour columns show '-'.")
