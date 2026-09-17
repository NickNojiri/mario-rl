# Wait for a PPO run's final checkpoint line, then evaluate latest.pt on train + held-out stages.
param(
    [string]$Run = "ppo_1h_a",
    [int]$Episodes = 10,
    [int]$Workers = 12
)
$log = "runs\$Run\stdout.log"
while (-not (Select-String -Path $log -Pattern "\[ckpt\] final" -Quiet)) {
    Start-Sleep -Seconds 20
}
Start-Sleep -Seconds 10  # let workers shut down and free the CPU
Write-Output "run finished; final eval started $(Get-Date -Format HH:mm:ss)"
wsl.exe -d Ubuntu --cd /mnt/c/Users/17143/mario-rl -- bash scripts/wsl_run.sh python -W ignore eval_stages.py `
    "runs/$Run/latest.pt" --stages all --episodes $Episodes --workers $Workers --noop-max 30 `
    --out "runs/$Run/eval_final_all.json" 2>$null
Write-Output "final eval done $(Get-Date -Format HH:mm:ss)"
