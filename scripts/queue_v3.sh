#!/usr/bin/env bash
# v3 experiment queue, fully unattended. Each experiment is the same length as ppo_1h_a so results compare
# directly. Then the best run (held-out mean x_pos) is continued for the remaining time.
#
#   b = new reward only            (death -150, hurt -50, new-ground progress, capped points)
#   c = b + enemy motion/position observation
#   d = c + full 12-button set
#
# Usage: bash scripts/queue_v3.sh [continue_hours]
set -uo pipefail
cd "$(dirname "$0")/.."
hours="${1:-4}"
log="runs/queue_v3.log"
mkdir -p runs
py() { bash scripts/wsl_run.sh python -u -W ignore "$@"; }
say() { echo "=== $* $(date '+%F %T')" | tee -a "$log"; }

say "queue start (continue best for ${hours}h)"

# Baseline with the new diagnostics (death causes, LEFT/NOOP use, points) so behaviour compares like for like.
py eval_stages.py runs/ppo_1h_a/latest_1h_frozen.pt --stages all --episodes 10 --workers 12 \
  --out runs/ppo_1h_a/eval_final_all_v3diag.json >> "$log" 2>&1
say "baseline ppo_1h_a re-evaluated"

for x in b c d; do
  run="v3${x}_1h"
  say "train $run"
  py train_ppo.py --preset "ppo_1h_v3${x}" --run-name "$run" >> "runs/$run.stdout.log" 2>&1
  say "eval $run"
  py eval_stages.py "runs/$run/latest.pt" --stages all --episodes 10 --workers 12 \
    --out "runs/$run/eval_final_all.json" >> "$log" 2>&1
  py -m scripts.compare_evals "runs/$run/eval_final_all.json" runs/ppo_1h_a/eval_final_all_v3diag.json \
    > "runs/$run/compare_vs_ppo_1h_a.txt" 2>&1
done

# Random baseline for the 12-button set (d); the 7-button random baseline already exists.
py eval_stages.py --random --config runs/v3d_1h/config.json --stages all --episodes 10 --workers 12 \
  --out runs/v3d_1h/eval_random_all.json >> "$log" 2>&1
for x in b c; do
  py -m scripts.compare_evals "runs/v3${x}_1h/eval_final_all.json" runs/ppo_1h_a/eval_random_all.json \
    > "runs/v3${x}_1h/compare_vs_random.txt" 2>&1
done
py -m scripts.compare_evals runs/v3d_1h/eval_final_all.json runs/v3d_1h/eval_random_all.json \
  > runs/v3d_1h/compare_vs_random.txt 2>&1
py -m scripts.v3_report > runs/v3_report.txt 2>&1
say "experiments done; report in runs/v3_report.txt"

# Continue the best v3 run (held-out mean x_pos) for the remaining hours.
best=$(py -m scripts.v3_report --best)
if [ -n "$best" ] && [ "$hours" != "0" ]; then
  steps=$(py -c "import json; print(json.load(open('runs/$best/config.json'))['total_steps'] + int($hours * 3600 * 500))")
  say "continue $best to $steps steps"
  py train_ppo.py --resume "runs/$best/latest.pt" --total-steps "$steps" --snapshot-every 2000000 \
    >> "runs/$best.stdout.log" 2>&1
  py eval_stages.py "runs/$best/latest.pt" --stages all --episodes 10 --workers 12 \
    --out "runs/$best/eval_long_all.json" >> "$log" 2>&1
  say "long continuation of $best evaluated"
fi
say "queue done"
