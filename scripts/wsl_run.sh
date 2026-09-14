#!/usr/bin/env bash
# Run a command inside the project venv (lives in WSL home, not on /mnt/c, for I/O speed).
# From Windows: wsl -d Ubuntu --cd /mnt/c/Users/17143/mario-rl -- bash scripts/wsl_run.sh python train.py --preset smoke
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/.venvs/mario-rl/bin/activate"
exec "$@"
