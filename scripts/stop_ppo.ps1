# SIGTERM -> train_ppo.py saves latest.pt, closes env workers and exits. Resume with start_ppo.ps1 -Resume.
wsl.exe -d Ubuntu -- pkill -TERM -f "^python -u -W ignore train_ppo.py"
Write-Output "stop signal sent; wait for '[ckpt] final' in runs\<name>\stdout.log"
