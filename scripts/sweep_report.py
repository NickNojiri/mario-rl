"""Rank sweep runs with uncertainty, and a low-noise training-progress metric.

    python -m scripts.sweep_report            # table
    python -m scripts.sweep_report --reeval   # first re-evaluate every run with 12 held-out episodes

held_out_x +- SE comes from the per-stage standard errors of the eval. train_ep_x is the mean x_pos over each
run's last 2000 non-practice training episodes: thousands of episodes, so its noise is small, but it measures
progress on stages the run trained on, not generalization.
"""
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

EVAL_NAME = "eval_test12.json"


def reeval():
    for run in sorted(Path("runs").glob("sweep_*")):
        if not (run / "latest.pt").exists() or (run / EVAL_NAME).exists():
            continue
        print(f"evaluating {run.name}", flush=True)
        subprocess.run(["bash", "scripts/wsl_run.sh", "python", "-W", "ignore", "eval_stages.py",
                        str(run / "latest.pt"), "--stages", "test", "--episodes", "12", "--workers", "12",
                        "--out", str(run / EVAL_NAME)], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)


def held_out(run: Path):
    path = run / EVAL_NAME if (run / EVAL_NAME).exists() else run / "eval_test.json"
    if not path.exists():
        return None, None, None
    d = json.load(open(path))
    per = [(r["mean_x_pos"], r["std_x_pos"] / math.sqrt(r["episodes"])) for r in d["stages"]]
    mean = sum(m for m, _ in per) / len(per)
    se = math.sqrt(sum(s * s for _, s in per)) / len(per)
    return mean, se, d["test_summary"]["flag_rate"]


def training(run: Path):
    rows = [e for e in csv.DictReader(open(run / "episodes.csv")) if e.get("practice") != "True"][-2000:]
    if not rows:
        return {}
    n = len(rows)
    steps = sum(int(e["length"]) for e in rows) or 1
    causes = Counter(e["death_cause"] for e in rows)
    return {"x": sum(int(e["x_pos"]) for e in rows) / n,
            "se": (sum((int(e["x_pos"]) - sum(int(f["x_pos"]) for f in rows) / n) ** 2 for e in rows) / n / n) ** 0.5,
            "flags": sum(e["flag_get"] == "True" for e in rows) / n,
            "pit": causes["pit"] / n,
            "enemy": sum(v for k, v in causes.items() if k.startswith("enemy")) / n,
            "stomps": sum(int(e["point_events"]) for e in rows) / n,
            "jumps": sum(int(e["jumps"]) for e in rows) / n,
            "left": sum(int(e["left_presses"]) for e in rows) / steps,
            "points": sum(int(e["points"]) for e in rows) / n}


def main():
    if "--reeval" in sys.argv:
        reeval()
    runs = []
    for run in sorted(Path("runs").glob("sweep_*")):
        if not (run / "episodes.csv").exists():
            continue
        mean, se, flags = held_out(run)
        if mean is None:
            continue
        runs.append((run.name.replace("sweep_", ""), mean, se, flags, training(run)))
    if not runs:
        return print("no sweep runs yet")
    base = next((r for r in runs if r[0] == "base"), None)
    runs.sort(key=lambda r: -r[1])
    print(f"{'variant':14} {'held-out x':>14} {'vs base':>8} {'flags':>6} | {'train_ep_x':>11} {'flags':>6} "
          f"{'pit':>5} {'enemy':>6} {'stomps':>7} {'jumps':>6} {'LEFT':>6} {'pts':>5}")
    for name, mean, se, flags, t in runs:
        delta = ""
        if base and name != "base":
            diff = mean - base[1]
            sigma = math.sqrt(se ** 2 + base[2] ** 2)
            delta = f"{diff:+.0f} ({diff / sigma:+.1f}s)" if sigma else ""
        print(f"{name:14} {mean:8.0f} +-{se:<4.0f} {delta:>8} {flags:6.1%} | {t.get('x', 0):8.0f} +-{t.get('se', 0):<2.0f} "
              f"{t.get('flags', 0):6.1%} {t.get('pit', 0):5.0%} {t.get('enemy', 0):6.0%} {t.get('stomps', 0):7.2f} "
              f"{t.get('jumps', 0):6.1f} {t.get('left', 0):6.1%} {t.get('points', 0):5.0f}")
    print("\n'vs base' is the difference in held-out x_pos with its size in standard errors (s); anything under "
          "about 2s is noise.\ntrain_ep_x: mean distance over the last 2000 training episodes (low noise, but "
          "measures the stages it trains on).")


if __name__ == "__main__":
    main()
