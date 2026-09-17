"""One line per eval JSON: python -m scripts.compare_summaries <label=path> [...]"""
import json
import sys
from pathlib import Path

for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    if not Path(path).exists():
        print(f"{label:22} (missing {path})")
        continue
    d = json.load(open(path))
    t, r = d.get("test_summary", {}), d.get("train_summary", {})
    print(f"{label:22} held-out x={t.get('mean_x_pos', 0):6.0f} flags={t.get('flag_rate', 0):5.1%} "
          f"pit={t.get('pit_death_rate', 0):4.0%} | train x={r.get('mean_x_pos', 0):6.0f} "
          f"flags={r.get('flag_rate', 0):5.1%}")
