#!/usr/bin/env bash
# Supervisor for a long run: keeps train_ppo.py going across crashes and evaluates every snapshot on the held-out
# stages as it appears. Start it from Windows with scripts/start_week.ps1 (safe to rerun, e.g. after a reboot).
# Usage: bash scripts/week_run.sh <run> <total_steps> <snapshot_every>
# Expects runs/<run>/latest.pt. Markers it leaves in runs/<run>:
#   DONE     training reached total_steps (a final all-stage evaluation follows)
#   STOPPED  training exited cleanly before total_steps (scripts/stop_ppo.ps1); nothing restarts it
#   GAVE_UP  3 crashes within 30 minutes; read stdout.log before restarting
#   ALERT.txt  the health check saw a non-finite loss and paused the run
# Pause/resume (scripts/pause.ps1, resume.ps1) work as usual; evaluations also wait while paused.
set -uo pipefail
cd "$(dirname "$0")/.."
run="$1"; total="$2"; snap="$3"
dir="runs/$run"; log="$dir/stdout.log"
py() { bash scripts/wsl_run.sh python -u -W ignore "$@"; }
rm -f "$dir/DONE" "$dir/STOPPED" "$dir/GAVE_UP"
mkdir -p "$dir/snapshots"

eval_pass() {
  # Every snapshot without a result, oldest first. Results land under a temp name and are renamed when complete,
  # so a kill mid-evaluation just means that snapshot is evaluated again next pass.
  for s in $(ls "$dir"/snapshots/step_*.pt 2>/dev/null | sort); do
    name=$(basename "$s" .pt); out="$dir/eval_${name}_test.json"
    [ -f "$out" ] && continue
    while [ -f "$dir/PAUSE" ]; do sleep 30; done
    if nice -n 19 bash scripts/wsl_run.sh python -W ignore eval_stages.py "$s" --stages test --episodes 10 \
        --workers 2 --out "$out.tmp" >> "$dir/eval.log" 2>&1; then
      mv "$out.tmp" "$out"
      echo "$name $(date '+%F %H:%M') $(py -c "import json,sys; t=json.load(open(sys.argv[1]))['test_summary']; \
print(f\"heldout_x={t['mean_x_pos']:.1f} flag_rate={t['flag_rate']:.3f}\")" "$out")" >> "$dir/heldout_curve.txt"
    fi
  done
}

watcher() {
  while :; do
    finished=0
    { [ -f "$dir/DONE" ] || [ -f "$dir/STOPPED" ] || [ -f "$dir/GAVE_UP" ]; } && finished=1
    eval_pass
    py scripts/run_health.py "$dir" >> "$dir/health.log" 2>&1
    if [ $? -eq 2 ] && [ ! -f "$dir/ALERT.txt" ]; then
      touch "$dir/PAUSE"
      { echo "$(date '+%F %H:%M') paused by the health check: non-finite loss."; tail -n 1 "$dir/health.log"
        echo "Resuming would continue from the diverged weights; roll latest.pt back to a snapshot first."
      } > "$dir/ALERT.txt"
    fi
    [ $finished -eq 1 ] && break
    sleep 300
  done
  if [ -f "$dir/DONE" ]; then
    py eval_stages.py "$dir/latest.pt" --stages all --episodes 10 --workers 12 --out "$dir/eval_final_all.json" \
      >> "$dir/eval.log" 2>&1
  fi
}

watcher &
watch_pid=$!

crash_times=()
while :; do
  echo "=== train start $(date '+%F %T')" >> "$log"
  py train_ppo.py --resume "$dir/latest.pt" --total-steps "$total" --snapshot-every "$snap" >> "$log" 2>&1
  code=$?
  if [ $code -eq 0 ]; then
    step=$(tail -n 20 "$log" | grep -o '\[ckpt\] final step=[0-9]*' | tail -1 | cut -d= -f2)
    if [ "${step:-0}" -ge "$total" ]; then touch "$dir/DONE"; else touch "$dir/STOPPED"; fi
    echo "=== train end $(date '+%F %T') step=${step:-?} ($( [ -f "$dir/DONE" ] && echo done || echo stopped))" >> "$log"
    break
  fi
  now=$(date +%s)
  recent=()
  for t in "${crash_times[@]}" "$now"; do [ $((now - t)) -lt 1800 ] && recent+=("$t"); done
  crash_times=("${recent[@]}")
  echo "=== train crashed with exit code $code at $(date '+%F %T') (${#crash_times[@]} in the last 30 min)" >> "$log"
  if [ ${#crash_times[@]} -ge 3 ]; then
    touch "$dir/GAVE_UP"
    echo "=== giving up: repeated crashes, see above" >> "$log"
    break
  fi
  sleep 60
done
wait "$watch_pid"
echo "=== supervisor exit $(date '+%F %T')" >> "$log"
