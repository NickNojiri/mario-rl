#!/usr/bin/env bash
# P4 baseline: gen2's last three checkpoints, standard protocol, current code. Produced the JSON files here.
cd "$(dirname "$0")/../../.."
for c in snapshots/step_012002304 snapshots/step_014002176 latest; do
  n=$(basename "$c")
  bash scripts/wsl_run.sh python -W ignore eval_stages.py "runs/gen2/$c.pt" --stages test --episodes 10 --workers 10 \
    --out "runs/gen2/p4_baseline_$n.json" 2>/dev/null | grep '^test:' | cut -c1-60
done
