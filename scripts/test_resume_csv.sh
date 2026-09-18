#!/usr/bin/env bash
# Resume a smoke run whose updates.csv header differs from the current schema; the old log must be kept aside.
set -euo pipefail
cd "$(dirname "$0")/.."
run=runs/smoke_csv
rm -rf "$run"
bash scripts/wsl_run.sh python -W ignore train_ppo.py --preset ppo_smoke --run-name smoke_csv > /dev/null 2>&1
sed -i '1s/wall_time/wall_time_OLD/' "$run/updates.csv"  # csv lines end in \r\n, so no $ anchor
bash scripts/wsl_run.sh python -W ignore train_ppo.py --resume "$run/latest.pt" --total-steps 2560 > /dev/null 2>&1
ls "$run"
echo "new header ends with: $(head -1 "$run/updates.csv" | awk -F, '{print $NF}')"
echo "kept old log: $(ls "$run" | grep -c 'updates_upto_')"
rm -rf "$run"
