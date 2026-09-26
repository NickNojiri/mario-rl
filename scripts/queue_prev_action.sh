#!/usr/bin/env bash
# Previous-action ablation, the controlled version of the experiment suggested in review.
#
#   6 runs: previous action visible / hidden, 3 seeds each, identical budget (2M steps) and settings.
#   Base config is ppo_ablate_2m: the v3b reward, no procgen, no PLR, so the arms differ in one thing only.
#   Each run is followed by the standard evaluation (10 episodes/stage, seeds 50000+, randomized start
#   offsets), which reports the three numbers asked for: stall rate, completion (flag) rate and distance.
#
#   bash scripts/queue_prev_action.sh          # ~6 h; skips runs that already finished
#
# Sequential on purpose: 12 envs already oversubscribe 8 physical cores, so running arms in parallel would
# make each one slower and add contention noise to a comparison that is about a small effect.
set -u
cd /mnt/c/Users/17143/mario-rl
mkdir -p runs

for seed in 0 1 2; do
    for prev in true false; do
        name="prevact_${prev}_s${seed}"
        if [ -f "runs/$name/eval_final_all.json" ]; then
            echo "[skip] $name already evaluated"
            continue
        fi
        echo "[train] $name  $(date +%H:%M)"
        bash scripts/wsl_run.sh python -u -W ignore train_ppo.py \
            --preset ppo_ablate_2m --run-name "$name" --seed "$seed" \
            --set "obs_prev_action=$prev" > "runs/${name}_train.log" 2>&1
        echo "[eval ] $name  $(date +%H:%M)"
        bash scripts/wsl_run.sh python -W ignore eval_stages.py "runs/$name/latest.pt" \
            --stages all --episodes 10 --workers 12 \
            --out "runs/$name/eval_final_all.json" > "runs/${name}_eval.log" 2>&1
    done
done

echo
echo "=== previous-action ablation: held-out ==="
bash scripts/wsl_run.sh python - <<'PY'
import json, glob, statistics
arms = {"true": [], "false": []}
for path in sorted(glob.glob("runs/prevact_*/eval_final_all.json")):
    arm = "true" if "_true_" in path else "false"
    t = json.load(open(path))["test_summary"]
    arms[arm].append((path.split("/")[1], t["mean_x_pos"], t["flag_rate"], t["stall_rate"]))
for arm, rows in arms.items():
    if not rows:
        continue
    print(f"\nprevious action {'VISIBLE' if arm == 'true' else 'HIDDEN '}")
    for name, x, flag, stall in rows:
        print(f"  {name:<22} x={x:7.1f}  flags={flag:.3f}  stall={stall:.3f}")
    xs = [r[1] for r in rows]
    spread = f" sd={statistics.stdev(xs):6.1f}" if len(xs) > 1 else ""
    print(f"  {'mean of seeds':<22} x={statistics.mean(xs):7.1f}{spread}  (n={len(xs)})")
PY
