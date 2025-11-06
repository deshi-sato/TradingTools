param(
  [ValidateSet("baseline","lstm","transformer")]$Mode="lstm",
  [string]$Symbols="6501,8035"
)
$root = "C:\Users\Owner\Documents\desshi_signal_viewer\deep_naut"
cd $root
.\.venv\Scripts\Activate.ps1
$log = ".\logs\runner_${Mode}_$(Get-Date -Format yyyyMMdd_HHmm).log"
py .\scripts\naut_runner_ml.py --mode $Mode --symbols $Symbols --log $log `
  --p-min 0.72 --ev-floor 0.02 --spread-max 0.0048
