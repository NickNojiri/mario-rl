# Pause a training run in place: it saves a checkpoint and idles (~0% CPU) until .\scripts\resume.ps1.
# Usage: .\scripts\pause.ps1 -Run week1
param([Parameter(Mandatory = $true)][string]$Run)
$dir = Join-Path $PSScriptRoot "..\runs\$Run"
if (-not (Test-Path $dir)) { Write-Output "no run folder: runs\$Run"; exit 1 }
New-Item -ItemType File -Force (Join-Path $dir "PAUSE") | Out-Null
Write-Output "pause requested for runs\$Run -- it stops after the current update (up to ~5 s)."
Write-Output "Check with: Get-Content runs\$Run\stdout.log -Tail 3   (look for [pause])"
