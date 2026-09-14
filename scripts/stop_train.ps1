# SIGTERM -> train.py saves latest.pt (+ buffer) and exits. Resume with start_train.ps1 -Resume.
wsl.exe -d Ubuntu -- pkill -TERM -f "^python -u -W ignore train.py"
Write-Output "stop signal sent; wait for '[ckpt] final' in runs\<name>\stdout.log"
