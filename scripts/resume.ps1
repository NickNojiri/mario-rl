# Resume a paused training run. It continues exactly where it stopped (same process, same game states).
# Usage: .\scripts\resume.ps1 -Run week1
param([Parameter(Mandatory = $true)][string]$Run)
$flag = Join-Path $PSScriptRoot "..\runs\$Run\PAUSE"
if (-not (Test-Path $flag)) { Write-Output "runs\$Run is not paused"; exit 0 }
$alert = Join-Path $PSScriptRoot "..\runs\$Run\ALERT.txt"
if (Test-Path $alert) { Get-Content $alert; Write-Output "not resuming: deal with ALERT.txt first, then delete it"; exit 1 }
Remove-Item $flag
Write-Output "resumed runs\$Run -- training continues within ~5 s."
