#!/usr/bin/env bash
# Runs after queue_v3.sh finishes its experiments: replaces its "continue best v3" step with the v4 experiment,
# then continues whichever run (v3b/c/d or v4) is best on held-out stages.
# Usage: bash scripts/queue_v4.sh [continue_hours]
set -uo pipefail
cd "$(dirname "$0")/.."
hours="${1:-3.5}"
log="runs/queue_v4.log"
py() { bash scripts/wsl_run.sh python -u -W ignore "$@"; }
say() { echo "=== $* $(date '+%F %T')" | tee -a "$log"; }

say "waiting for v3 experiments to finish"
until grep -q "experiments done" runs/queue_v3.log 2>/dev/null; do sleep 30; done
# Stop queue_v3 before (or just as) it starts continuing its best run; retry briefly to win any race.
for _ in $(seq 1 12); do
  pkill -f "bash scripts/queue_v3.sh" 2>/dev/null
  pkill -TERM -f "train_ppo.py --resume runs/v3" 2>/dev/null
  sleep 5
done
say "v3 queue stopped; starting v4"

run="v4_1h"
py train_ppo.py --preset ppo_1h_v4 --run-name "$run" >> "runs/$run.stdout.log" 2>&1
say "eval $run"
for m in safe insane; do
  py eval_stages.py "runs/$run/latest.pt" --stages all --episodes 10 --workers 12 --mode "$m" \
    --out "runs/$run/eval_final_all_$m.json" >> "$log" 2>&1
done
py eval_stages.py --random --config "runs/$run/config.json" --stages all --episodes 10 --workers 12 \
  --out "runs/$run/eval_random_all.json" >> "$log" 2>&1
py -m scripts.compare_evals "runs/$run/eval_final_all_safe.json" runs/ppo_1h_a/eval_final_all_v3diag.json \
  > "runs/$run/compare_safe_vs_ppo_1h_a.txt" 2>&1
py -m scripts.compare_evals "runs/$run/eval_final_all_safe.json" "runs/$run/eval_random_all.json" \
  > "runs/$run/compare_safe_vs_random.txt" 2>&1

say "pit forensics"
mkdir -p runs/analysis
best_v3=$(py -m scripts.v3_report --best-v3)
py -m scripts.pit_forensics "runs/$run/latest.pt" --episodes 6 --mode safe > /dev/null 2>&1
[ -n "$best_v3" ] && py -m scripts.pit_forensics "runs/$best_v3/latest.pt" --episodes 6 > /dev/null 2>&1
py -m scripts.v3_report > runs/v3_v4_report.txt 2>&1
say "report in runs/v3_v4_report.txt"

best=$(py -m scripts.v3_report --best)
if [ -n "$best" ] && [ "$hours" != "0" ]; then
  steps=$(py -c "import torch; print(torch.load('runs/$best/latest.pt', map_location='cpu', weights_only=False)['global_step'] + int($hours * 3600 * 480))")
  say "continue $best to $steps steps"
  py train_ppo.py --resume "runs/$best/latest.pt" --total-steps "$steps" --snapshot-every 2000000 \
    >> "runs/$best.stdout.log" 2>&1
  py eval_stages.py "runs/$best/latest.pt" --stages all --episodes 10 --workers 12 \
    --out "runs/$best/eval_long_all.json" >> "$log" 2>&1
  say "long continuation of $best evaluated"
fi
say "queue done"
