param(
  [ValidateSet('DRYRUN','PAPER','LIVE')][string]$Mode='PAPER',
  [switch]$Shutdown
)

$env:MODE = $Mode
py .\scripts\closeout_10am.py

# 直近のサマリを表示
Get-Content (Get-ChildItem .\logs\close_summary-*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName

if ($Shutdown) {
  Write-Host "Shutting down Windows..." -ForegroundColor Yellow
  shutdown /s /t 0
}
