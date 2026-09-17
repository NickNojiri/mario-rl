# Launch a detached PPO run in WSL that survives closing the terminal / Claude session.
# Usage: .\scripts\start_ppo.ps1 -RunName ppo_1h_a -Preset ppo_1h
#        .\scripts\start_ppo.ps1 -Resume runs/ppo_1h_a/latest.pt
# Stop cleanly (saves checkpoint): .\scripts\stop_ppo.ps1
param(
    [string]$RunName = "ppo_1h_a",
    [string]$Preset = "ppo_1h",
    [string]$Resume = ""
)
if ($Resume) {
    $logName = Split-Path (Split-Path $Resume -Parent) -Leaf
    $trainArgs = "--resume $Resume"
} else {
    $logName = $RunName
    $trainArgs = "--preset $Preset --run-name $RunName"
}
# Start-Process joins arguments with spaces, so every argument stays space-free.
Start-Process wsl.exe -WindowStyle Hidden -ArgumentList (
    "-d Ubuntu --cd /mnt/c/Users/17143/mario-rl -- bash scripts/train_ppo_detached.sh $logName $trainArgs"
)
Write-Output "started: train_ppo.py $trainArgs  (log: runs\$logName\stdout.log)"
