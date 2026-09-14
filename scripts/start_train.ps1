# Launch a detached training run in WSL that survives closing the terminal / Claude session.
# Usage: .\scripts\start_train.ps1 -RunName dqn_1_1            (fresh full run)
#        .\scripts\start_train.ps1 -Resume runs/dqn_1_1/latest.pt
# Stop cleanly (saves checkpoint): .\scripts\stop_train.ps1
param(
    [string]$RunName = "dqn_1_1",
    [string]$Resume = ""
)
if ($Resume) {
    $logName = Split-Path (Split-Path $Resume -Parent) -Leaf
    $trainArgs = "--resume $Resume"
} else {
    $logName = $RunName
    $trainArgs = "--preset full --run-name $RunName"
}
# Start-Process joins arguments with spaces, so keep every argument space-free and let the .sh do the work.
# The hidden wsl.exe window keeps the WSL VM alive for the life of the run.
Start-Process wsl.exe -WindowStyle Hidden -ArgumentList (
    "-d Ubuntu --cd /mnt/c/Users/17143/mario-rl -- bash scripts/train_detached.sh $logName $trainArgs"
)
Write-Output "started: train.py $trainArgs  (log: runs\$logName\stdout.log)"
