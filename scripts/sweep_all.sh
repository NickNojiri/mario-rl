#!/usr/bin/env bash
# Run every sweep stage back to back, then write the ranked table. Usage: bash scripts/sweep_all.sh [stages]
set -uo pipefail
cd "$(dirname "$0")/.."
stages="${1:-a b c}"
mkdir -p runs/sweep
for s in $stages; do
  echo "=== stage $s $(date '+%F %T')" | tee -a runs/sweep/sweep.log
  bash scripts/wsl_run.sh python -u -W ignore -m scripts.sweep --stage "$s" >> runs/sweep/sweep.log 2>&1
done
bash scripts/wsl_run.sh python -W ignore -m scripts.sweep --show > runs/sweep/report.txt 2>&1
echo "=== sweep done $(date '+%F %T')" | tee -a runs/sweep/sweep.log
