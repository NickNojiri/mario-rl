#!/usr/bin/env bash
# Called by scripts/start_ppo.ps1. Args: <run name> <train_ppo.py args...>
set -euo pipefail
cd "$(dirname "$0")/.."
log="runs/$1/stdout.log"
shift
mkdir -p "$(dirname "$log")"
exec bash scripts/wsl_run.sh python -u -W ignore train_ppo.py "$@" >> "$log" 2>&1
