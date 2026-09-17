# Wait until run has at least $Index snapshots, then evaluate the newest one on held-out stages.
# Usage: .\scripts\eval_snapshot_when_ready.ps1 -Run ppo_1h_a -Index 1
param(
    [string]$Run = "ppo_1h_a",
    [int]$Index = 1,
    [int]$Episodes = 5,
    [int]$Workers = 3
)
$dir = "runs\$Run\snapshots"
while (-not (Test-Path $dir) -or @(Get-ChildItem $dir -Filter *.pt).Count -lt $Index) {
    Start-Sleep -Seconds 20
}
Start-Sleep -Seconds 5  # let torch.save finish
$snap = @(Get-ChildItem $dir -Filter *.pt | Sort-Object Name)[$Index - 1]
$rel = "runs/$Run/snapshots/$($snap.Name)"
$out = "runs/$Run/eval_$($snap.BaseName)_test.json"
Write-Output "evaluating $rel at $(Get-Date -Format HH:mm:ss)"
wsl.exe -d Ubuntu --cd /mnt/c/Users/17143/mario-rl -- bash scripts/wsl_run.sh python -W ignore eval_stages.py `
    $rel --stages test --episodes $Episodes --workers $Workers --noop-max 30 --out $out 2>$null
Write-Output "done at $(Get-Date -Format HH:mm:ss)"
