#!/usr/bin/env bash
# Continue a PPO run for a long stretch, then evaluate it automatically (no CPU contention while training).
# Usage: bash scripts/long_run_then_eval.sh <run_name> <total_steps> <snapshot_every>
# Expects runs/<run_name>/latest.pt to exist (copied from the run being continued).
set -uo pipefail
cd "$(dirname "$0")/.."
run="$1"; total="$2"; snap="$3"
dir="runs/$run"
log="$dir/stdout.log"
py() { bash scripts/wsl_run.sh python -u -W ignore "$@"; }

echo "=== train start $(date)" >> "$log"
py train_ppo.py --resume "$dir/latest.pt" --total-steps "$total" --snapshot-every "$snap" >> "$log" 2>&1
echo "=== train end $(date)" >> "$log"

# Held-out curve: every snapshot, 10 episodes per held-out stage.
for s in "$dir"/snapshots/*.pt; do
  name=$(basename "$s" .pt)
  py eval_stages.py "$s" --stages test --episodes 10 --workers 12 --out "$dir/eval_${name}_test.json" >> "$dir/eval.log" 2>&1
done
# Final: all stages, same protocol as the 1-hour run, plus behaviour diagnostics.
py eval_stages.py "$dir/latest.pt" --stages all --episodes 10 --workers 12 --out "$dir/eval_final_all.json" >> "$dir/eval.log" 2>&1
py -m scripts.compare_evals "$dir/eval_final_all.json" runs/ppo_1h_a/eval_random_all.json > "$dir/compare_vs_random.txt" 2>&1
py -m scripts.compare_evals "$dir/eval_final_all.json" runs/ppo_1h_a/eval_final_all.json > "$dir/compare_vs_1h.txt" 2>&1
py -m scripts.diagnose_actions "$dir/latest.pt" --episodes 4 > "$dir/diagnose_actions.txt" 2>&1
py -m scripts.diagnose_actions runs/ppo_1h_a/latest_1h_frozen.pt --episodes 4 > "$dir/diagnose_actions_1h.txt" 2>&1
py -m scripts.summarize_ppo "$dir" 200 > "$dir/summary.txt" 2>&1
echo "=== eval end $(date)" >> "$log"
