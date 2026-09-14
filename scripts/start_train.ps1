# Launch a detached training run in WSL that survives closing the terminal / Claude session.
# Usage: .\scripts\start_train.ps1 -RunName dqn_1_1            (fresh full run)
#        .\scripts\start_train.ps1 -Resume runs/dqn_1_1/latest.pt
# Stop cleanly (saves checkpoint): .\scripts\stop_train.ps1
param(
    [string]$RunName = "dqn_1_1",
    [string]$Resume = ""
)
$trainArgs = if ($Resume) { "--resume $Resume" } else { "--preset full --run-name $RunName" }
$logName = if ($Resume) { Split-Path (Split-Path $Resume -Parent) -Leaf } else { $RunName }
New-Item -ItemType Directory -Force "runs\$logName" | Out-Null
# The hidden wsl.exe window keeps the WSL VM alive for the life of the run.
Start-Process wsl.exe -WindowStyle Hidden -ArgumentList @(
    "-d", "Ubuntu", "--cd", "/mnt/c/Users/17143/mario-rl", "--",
    "bash", "-c", "bash scripts/wsl_run.sh python -u -W ignore train.py $trainArgs --save-buffer >> runs/$logName/stdout.log 2>&1"
)
Write-Output "started: train.py $trainArgs  (log: runs\$logName\stdout.log)"
