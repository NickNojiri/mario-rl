#!/usr/bin/env bash
# Called by scripts/start_train.ps1. Args: <log dir name> <train.py args...>
set -euo pipefail
cd "$(dirname "$0")/.."
log="runs/$1/stdout.log"
shift
mkdir -p "$(dirname "$log")"
exec bash scripts/wsl_run.sh python -u -W ignore train.py "$@" --save-buffer >> "$log" 2>&1
