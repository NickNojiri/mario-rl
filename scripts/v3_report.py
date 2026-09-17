"""Side-by-side table of the v3/v4 experiments vs the 1-hour v2 baseline and random, plus pit forensics.

    python -m scripts.v3_report            # table
    python -m scripts.v3_report --best     # run name with the best held-out mean x_pos (v3 runs and v4 safe mode)
    python -m scripts.v3_report --best-v3  # same, v3 runs only
"""
import json
import sys
from pathlib import Path

RUNS = [("ppo_1h_a (v2 baseline)", "ppo_1h_a", "runs/ppo_1h_a/eval_final_all_v3diag.json"),
        ("v3b: new reward", "v3b_1h", "runs/v3b_1h/eval_final_all.json"),
        ("v3c: + enemy motion", "v3c_1h", "runs/v3c_1h/eval_final_all.json"),
        ("v3d: + 12 buttons", "v3d_1h", "runs/v3d_1h/eval_final_all.json"),
        ("v4 SAFE mode", "v4_1h", "runs/v4_1h/eval_final_all_safe.json"),
        ("v4 INSANE mode", "v4_1h", "runs/v4_1h/eval_final_all_insane.json"),
        ("random (7 buttons)", None, "runs/ppo_1h_a/eval_random_all.json"),
        ("random (v4 buttons)", None, "runs/v4_1h/eval_random_all.json")]
KEYS = [("mean_x_pos", "x_pos", "{:.0f}"), ("flag_rate", "flags", "{:.1%}"), ("pit_death_rate", "pit", "{:.0%}"),
        ("enemy_death_rate", "enemy", "{:.0%}"), ("stall_rate", "stall", "{:.0%}"), ("left_share", "LEFT", "{:.1%}"),
        ("noop_share", "NOOP", "{:.1%}"), ("points_per_episode", "points", "{:.0f}")]

rows = []
for label, run, path in RUNS:
    if Path(path).exists():
        d = json.load(open(path))
        rows.append((label, run, d.get("train_summary", {}), d.get("test_summary", {})))

if "--best" in sys.argv or "--best-v3" in sys.argv:
    only_v3 = "--best-v3" in sys.argv
    cands = [(r[3].get("mean_x_pos", 0), r[1]) for r in rows
             if r[1] and (r[1].startswith("v3") or (not only_v3 and r[0] == "v4 SAFE mode"))]
    print(max(cands)[1] if cands else "")
    sys.exit()

for group, idx in (("HELD-OUT stages", 3), ("TRAINING stages", 2)):
    print(f"\n{group}")
    print(f"{'run':26}" + "".join(f"{h:>9}" for _, h, _ in KEYS))
    for r in rows:
        s = r[idx]
        print(f"{r[0]:26}" + "".join(f"{(fmt.format(s[k]) if k in s else '-'):>9}" for k, _, fmt in KEYS))

print("\nPIT FORENSICS (why pit deaths happen)")
for f in sorted(Path("runs/analysis").glob("pit_forensics_*.json")):
    d = json.load(open(f))
    total = d["pit_deaths"] or 1
    reasons = ", ".join(f"{k} {v / total:.0%}" for k, v in d["pit_death_reasons"].items())
    print(f"{f.stem.replace('pit_forensics_', ''):22} {d['pit_deaths']:3d} pit deaths in {d['episodes']} episodes: {reasons}")
print("\nNote: the 7-button random baseline predates the behaviour diagnostics, so those columns show '-'.")
