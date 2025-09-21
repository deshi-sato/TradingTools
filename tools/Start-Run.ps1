param([ValidateSet("DRYRUN","PAPER","LIVE")][string]$Mode="PAPER",[int]$Minutes=20)

$ErrorActionPreference = 'Stop'

# --- スクリプト自身の場所を確実に特定（$PSScriptRoot が null でも動く） ---
$scriptPath = $MyInvocation.MyCommand.Path
$scriptDir  = Split-Path -Parent $scriptPath
$proj       = Resolve-Path (Join-Path $scriptDir '..')

# --- タイムスタンプは必ず文字列に。念のためフォールバックも用意 ---
$ts = (Get-Date).ToString('yyyyMMdd-HHmmss')
if (-not $ts) { $ts = [DateTime]::Now.Ticks.ToString() }

# --- ログ出力先 ---
$logDir = Join-Path $proj 'logs\tasks'
New-Item $logDir -ItemType Directory -Force | Out-Null
$log = Join-Path $logDir ("intraday-{0}.log" -f $ts)
Start-Transcript -Path $log -Append | Out-Null

Write-Host "PWD       = $PWD"
Write-Host "ScriptDir = $scriptDir"
Write-Host "Project   = $proj"
Write-Host "PSVersion = $($PSVersionTable.PSVersion)"
Write-Host "PATH      = $env:PATH"

# --- 環境変数セット ---
$env:MODE        = $Mode
$env:RUN_MINUTES = "$Minutes"

# 週末ガード（PAPER かつ土日なら FAKE に切替）
$w = (Get-Date).DayOfWeek
$weekend = ($w -eq 'Saturday' -or $w -eq 'Sunday')
if ($Mode -eq 'DRYRUN') {
  $env:USE_FAKE_TICKS = '1'
}
elseif ($Mode -eq 'PAPER' -and $weekend) {
  $env:USE_FAKE_TICKS = '1'
  Write-Host "[guard] Weekend detected -> USE_FAKE_TICKS=1" -ForegroundColor Yellow
}
else {
  $env:USE_FAKE_TICKS = '0'
}

# 実API時のみレート制限対策でポーリングを少し長めに
if ($env:USE_FAKE_TICKS -eq '0') { $env:POLL_MS = '750' }

# --- venv の python.exe を絶対パスで実行（PATH に依存しない） ---
$py = Join-Path $proj 'venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "python not found: $py" }

& $py -m orchestrate.run_intraday
$code = $LASTEXITCODE
Write-Host "python exit code: $code"

Stop-Transcript | Out-Null
exit $code
