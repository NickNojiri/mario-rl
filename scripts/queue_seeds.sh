#!/usr/bin/env bash
# M1: multi-seed replication. Every conclusion in this repo rests on one training seed, and the reported error
# bars are episode-level variance *within* that seed, which understates the real thing.
#
#   bash scripts/queue_seeds.sh          # ~3 h; skips seeds already evaluated
#
# The prediction is pre-registered in README ("P1"): held-out mean x_pos beats the random baseline of 382 in at
# least 3 of 3 seeds. Do not edit that prediction after reading these results.
#
# Sequential on purpose: 12 envs already oversubscribe 8 physical cores. Do not start this while another
# training queue is running.
set -u
cd /mnt/c/Users/17143/mario-rl
mkdir -p runs docs/results/seeds

for seed in 1 2 3; do
    name="seed_s${seed}"
    if [ -f "runs/$name/eval_final_all.json" ]; then
        echo "[skip] $name already evaluated"
        continue
    fi
    echo "[train] $name  $(date +%H:%M)"
    bash scripts/wsl_run.sh python -u -W ignore train_ppo.py \
        --preset ppo_1h --run-name "$name" --seed "$seed" > "runs/${name}_train.log" 2>&1
    echo "[eval ] $name  $(date +%H:%M)"
    bash scripts/wsl_run.sh python -W ignore eval_stages.py "runs/$name/latest.pt" \
        --stages all --episodes 10 --workers 12 \
        --out "runs/$name/eval_final_all.json" > "runs/${name}_eval.log" 2>&1
    cp "runs/$name/eval_final_all.json" "docs/results/seeds/${name}_eval_final_all.json"
done

echo
echo "=== M1 multi-seed replication: held-out ==="
bash scripts/wsl_run.sh python - <<'PY'
import glob, json, statistics

RANDOM_BASELINE = 382  # same protocol, see README
rows = []
for path in sorted(glob.glob("runs/seed_s*/eval_final_all.json")):
    t = json.load(open(path))["test_summary"]
    rows.append((path.split("/")[1], t["mean_x_pos"], t["flag_rate"], t["stall_rate"]))

for name, x, flag, stall in rows:
    beats = "beats random" if x > RANDOM_BASELINE else "DOES NOT beat random"
    print(f"  {name:<10} held-out x={x:7.1f}  flags={flag:.3f}  stall={stall:.3f}   {beats}")

if len(rows) > 1:
    xs = [r[1] for r in rows]
    print(f"\n  across seeds: mean={statistics.mean(xs):.1f}  sd={statistics.stdev(xs):.1f}  "
          f"min={min(xs):.1f}  max={max(xs):.1f}  n={len(xs)}")
    held = sum(x > RANDOM_BASELINE for x in xs)
    print(f"  P1 prediction (>{RANDOM_BASELINE} in 3 of 3): {held} of {len(xs)} -- "
          f"{'HELD' if held == len(xs) else 'FALSIFIED'}")
PY
