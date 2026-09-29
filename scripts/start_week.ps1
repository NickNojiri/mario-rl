# Start the long run's supervisor (training with crash restarts + held-out evaluation of every snapshot).
# Safe to run again at any time, e.g. after a reboot: it does nothing if the run is already going, and otherwise
# resumes from runs\<Run>\latest.pt.
# Usage: .\scripts\start_week.ps1                      (week1, 300M steps, snapshot every 10M)
#        .\scripts\start_week.ps1 -Run week1 -TotalSteps 400000000
# Check on it: .\scripts\week_status.ps1   Pause: .\scripts\pause.ps1 -Run week1   Stop: .\scripts\stop_ppo.ps1
param(
    [string]$Run = "week1",
    [long]$TotalSteps = 300000000,
    [long]$SnapshotEvery = 10000000
)
$running = wsl.exe -d Ubuntu -e pgrep -f "week_run[.]sh $Run "
if ($running) { Write-Output "runs\$Run is already running (supervisor pid $running)"; exit 0 }
if (-not (Test-Path "runs\$Run\latest.pt")) { Write-Output "no runs\$Run\latest.pt to resume from"; exit 1 }
foreach ($m in "GAVE_UP", "ALERT.txt") {
    if (Test-Path "runs\$Run\$m") { Write-Output "runs\$Run\$m exists: read it (and stdout.log) first, then delete it"; exit 1 }
}
# Start-Process joins arguments with spaces, so every argument stays space-free.
Start-Process wsl.exe -WindowStyle Hidden -ArgumentList (
    "-d Ubuntu --cd /mnt/c/Users/17143/mario-rl -- bash scripts/week_run.sh $Run $TotalSteps $SnapshotEvery"
)
Write-Output "started runs\$Run to $TotalSteps steps (snapshot every $SnapshotEvery). Log: runs\$Run\stdout.log"
