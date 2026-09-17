# Start the live browser demo detached (keeps running after the terminal closes), then open it.
# Usage: .\scripts\start_demo.ps1                 (held-out stages, looping)
#        .\scripts\start_demo.ps1 -Stages 1-1,3-3
# Stop:  .\scripts\stop_demo.ps1
param(
    [string]$Stages = "test",
    [string]$Checkpoint = "runs/ppo_1h_a/latest.pt",
    [int]$Port = 8765
)
Start-Process wsl.exe -WindowStyle Hidden -ArgumentList (
    "-d Ubuntu --cd /mnt/c/Users/17143/mario-rl -- bash scripts/wsl_run.sh python -u -W ignore -m scripts.demo " +
    "--checkpoint $Checkpoint --stages $Stages --port $Port"
)
Start-Sleep -Seconds 6
Start-Process "http://localhost:$Port"
Write-Output "demo running at http://localhost:$Port  (stop with .\scripts\stop_demo.ps1)"
