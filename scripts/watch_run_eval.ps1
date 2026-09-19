# Early-warning monitor: evaluate each new snapshot of a run on held-out stages as it appears, then do the
# final evaluation and head-to-head comparisons when training ends.
# Usage: .\scripts\watch_run_eval.ps1 -Run gen2
param(
    [string]$Run = "gen2",
    [int]$SnapshotEpisodes = 6,
    [int]$Workers = 3
)
$done = @{}
$log = "runs\$Run\heldout_curve.txt"
function Eval-Snapshot($snap) {
    $rel = "runs/$Run/snapshots/$($snap.Name)"
    $out = "runs/$Run/eval_$($snap.BaseName)_test.json"
    wsl.exe -d Ubuntu --cd /mnt/c/Users/17143/mario-rl -- bash scripts/wsl_run.sh python -W ignore eval_stages.py `
        $rel --stages test --episodes $SnapshotEpisodes --workers $Workers --out $out 2>$null |
        Select-String "^test:" | ForEach-Object { "$($snap.BaseName) $(Get-Date -Format HH:mm) $($_.Line)" } |
        Tee-Object -FilePath $log -Append
}
while (-not (Select-String -Path "runs\$Run\stdout.log" -Pattern "\[ckpt\] final" -Quiet -ErrorAction SilentlyContinue)) {
    Get-ChildItem "runs\$Run\snapshots" -Filter *.pt -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object {
        if (-not $done.ContainsKey($_.Name)) { Start-Sleep 5; Eval-Snapshot $_; $done[$_.Name] = $true }
    }
    Start-Sleep 60
}
Start-Sleep 15
wsl.exe -d Ubuntu --cd /mnt/c/Users/17143/mario-rl -- bash scripts/wsl_run.sh python -W ignore eval_stages.py `
    "runs/$Run/latest.pt" --stages all --episodes 10 --workers 12 --out "runs/$Run/eval_final_all.json" 2>$null | Out-Null
wsl.exe -d Ubuntu --cd /mnt/c/Users/17143/mario-rl -- bash scripts/wsl_run.sh python -m scripts.compare_summaries `
    "random=runs/ppo_1h_a/eval_random_all.json" "v3b_7.8M=runs/v3b_1h/eval_long_all.json" `
    "gen1_7.8M=runs/gen_1h/eval_long_all.json" "$Run=runs/$Run/eval_final_all.json"
wsl.exe -d Ubuntu --cd /mnt/c/Users/17143/mario-rl -- bash scripts/wsl_run.sh python -m scripts.compare_evals `
    "runs/$Run/eval_final_all.json" runs/v3b_1h/eval_long_all.json | Select-String "test|train:"
Get-Content $log
