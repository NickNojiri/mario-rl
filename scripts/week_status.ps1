# One-screen status of a long run: is it running, where is it, and the held-out curve so far.
# Usage: .\scripts\week_status.ps1 [-Run week1]
param([string]$Run = "week1")
$dir = "runs\$Run"
$running = wsl.exe -d Ubuntu -e pgrep -f "week_run[.]sh $Run "
Write-Output ("supervisor: " + $(if ($running) { "running" } else { "not running" }))
foreach ($m in "PAUSE", "DONE", "STOPPED", "GAVE_UP", "ALERT.txt") {
    if (Test-Path "$dir\$m") { Write-Output "marker: $m" }
}
if (Test-Path "$dir\ALERT.txt") { Get-Content "$dir\ALERT.txt" }
if (Test-Path "$dir\health.log") { Write-Output ("last health check: " + (Get-Content "$dir\health.log" -Tail 1)) }
Write-Output "--- held-out curve (10 episodes per held-out stage; gen2 at 12M/14M/15M: 506.6 / 718.9 / 670.5)"
if (Test-Path "$dir\heldout_curve.txt") { Get-Content "$dir\heldout_curve.txt" } else { Write-Output "(no snapshots evaluated yet)" }
Write-Output "--- last log lines"
Get-Content "$dir\stdout.log" -Tail 200 | Select-String "^(===|\[pause\]|\[resume\]|resumed|interrupted|Traceback)" |
    Select-Object -Last 5 | ForEach-Object { $_.Line }
