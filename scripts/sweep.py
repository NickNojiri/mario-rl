"""Parameter sweep: train short runs from one base config, evaluate each the same way, rank them.

    python -m scripts.sweep --stage a            # learning knobs
    python -m scripts.sweep --stage b            # reward knobs
    python -m scripts.sweep --stage c            # v4 feature toggles
    python -m scripts.sweep --show               # table of everything run so far

Objective: held-out mean x_pos (5 stages never trained on), flag rate as the tiebreak. Every variant trains
the same number of steps from scratch, then gets the same evaluation, so the numbers are comparable.
Behaviour columns come from the last 2000 training episodes of each run.
"""
import argparse
import csv
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

RESULTS = Path("runs/sweep/results.csv")
FIELDS = ["variant", "overrides", "steps", "heldout_x", "heldout_flags", "heldout_pit", "train_x", "train_flags",
          "pit_rate", "enemy_rate", "stomps_per_ep", "jumps_per_ep", "left_share", "points_per_ep", "minutes"]

STAGES = {
    # base = ppo_sweep preset (v3b reward). Each variant changes one thing.
    "a": {  # learning knobs
        "base": [],
        "lr_low": ["lr=0.0001"], "lr_high": ["lr=0.0006"],
        "ent_low": ["ent_coef=0.003"], "ent_high": ["ent_coef=0.03"],
        "gamma_995": ["gamma=0.995"],
        "rollout_512": ["rollout_len=512", "minibatches=12"],
    },
    "b": {  # reward shape
        "death_50": ["death_reward=-50"], "death_400": ["death_reward=-400"],
        "points_0": ["points_per_100=0"], "points_15": ["points_per_100=15", "points_cap=300"],
        "flag_600": ["flag_reward=600"],
        "stall_100": ["no_progress_steps=100"],
    },
    "c": {  # v4 features, one at a time, no modes
        "macros": ["actions=simple_macro"],
        "hints": ["obs_version=4"],
        "enemy_obs": ["obs_version=3"],
        "practice": ["practice_prob=0.25"],
        "plr": ["plr=true"],
    },
}


def run_variant(name: str, overrides: list[str], episodes_test: int, episodes_train: int) -> dict:
    run = f"sweep_{name}"
    run_dir = Path("runs") / run
    t0 = time.time()
    if not (run_dir / "latest.pt").exists():
        subprocess.run(["bash", "scripts/wsl_run.sh", "python", "-u", "-W", "ignore", "train_ppo.py",
                        "--preset", "ppo_sweep", "--run-name", run, *(["--set"] + overrides if overrides else [])],
                       check=True, stdout=open(f"runs/{run}.stdout.log", "w"), stderr=subprocess.STDOUT)
    evals = {}
    for group, eps in (("test", episodes_test), ("train", episodes_train)):
        out = run_dir / f"eval_{group}.json"
        if not out.exists():
            subprocess.run(["bash", "scripts/wsl_run.sh", "python", "-W", "ignore", "eval_stages.py",
                            str(run_dir / "latest.pt"), "--stages", group, "--episodes", str(eps),
                            "--workers", "12", "--out", str(out)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        evals[group] = json.load(open(out))[f"{group}_summary"]

    eps_rows = list(csv.DictReader(open(run_dir / "episodes.csv")))
    eps_rows = [e for e in eps_rows if e.get("practice") != "True"][-2000:]
    n = max(len(eps_rows), 1)
    steps = sum(int(e["length"]) for e in eps_rows) or 1
    causes = Counter(e["death_cause"] for e in eps_rows)
    return {
        "variant": name, "overrides": " ".join(overrides), "steps": eps_rows[-1]["step"] if eps_rows else 0,
        "heldout_x": round(evals["test"]["mean_x_pos"], 1), "heldout_flags": round(evals["test"]["flag_rate"], 4),
        "heldout_pit": round(evals["test"]["pit_death_rate"], 3),
        "train_x": round(evals["train"]["mean_x_pos"], 1), "train_flags": round(evals["train"]["flag_rate"], 4),
        "pit_rate": round(causes["pit"] / n, 3),
        "enemy_rate": round(sum(v for k, v in causes.items() if k.startswith("enemy")) / n, 3),
        "stomps_per_ep": round(sum(int(e["point_events"]) for e in eps_rows) / n, 2),
        "jumps_per_ep": round(sum(int(e["jumps"]) for e in eps_rows) / n, 1),
        "left_share": round(sum(int(e["left_presses"]) for e in eps_rows) / steps, 4),
        "points_per_ep": round(sum(int(e["points"]) for e in eps_rows) / n, 1),
        "minutes": round((time.time() - t0) / 60, 1),
    }


def show():
    if not RESULTS.exists():
        return print("no results yet")
    rows = sorted(csv.DictReader(open(RESULTS)), key=lambda r: -float(r["heldout_x"]))
    print(f"{'variant':14} {'held-out x':>10} {'flags':>6} {'pit':>5} {'train x':>8} {'stomps':>7} {'jumps':>6} "
          f"{'LEFT':>6} {'pts':>5}  overrides")
    for r in rows:
        print(f"{r['variant']:14} {float(r['heldout_x']):10.0f} {float(r['heldout_flags']):6.1%} "
              f"{float(r['heldout_pit']):5.0%} {float(r['train_x']):8.0f} {float(r['stomps_per_ep']):7.2f} "
              f"{float(r['jumps_per_ep']):6.1f} {float(r['left_share']):6.1%} {float(r['points_per_ep']):5.0f}  "
              f"{r['overrides']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=list(STAGES))
    p.add_argument("--only", help="comma list of variant names within the stage")
    p.add_argument("--episodes-test", type=int, default=12)  # SE ~25 on held-out mean x_pos (6 gave ~36)
    p.add_argument("--episodes-train", type=int, default=3)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    if args.show or not args.stage:
        return show()

    variants = STAGES[args.stage]
    if args.only:
        variants = {k: v for k, v in variants.items() if k in args.only.split(",")}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    done = {r["variant"] for r in csv.DictReader(open(RESULTS))} if RESULTS.exists() else set()
    new = not RESULTS.exists()
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for name, overrides in variants.items():
            if name in done:
                print(f"skip {name} (already in results)", flush=True)
                continue
            print(f"=== {name}: {' '.join(overrides) or 'base'} {time.strftime('%H:%M:%S')}", flush=True)
            row = run_variant(name, overrides, args.episodes_test, args.episodes_train)
            w.writerow(row)
            f.flush()
            print({k: row[k] for k in ("heldout_x", "heldout_flags", "train_x", "pit_rate", "stomps_per_ep")},
                  flush=True)
    show()


if __name__ == "__main__":
    main()
