param(
  [ValidateSet(''buy'',''sell'',''price'',''run'')][string]$Action,
  [string]$Symbol,
  [int]$Qty = 100,
  [double]$Price,
  [switch]$DryRun
)
$ErrorActionPreference = ''Stop''
Set-StrictMode -Version Latest

if (-not $env:KABU_BASE_URL) { $env:KABU_BASE_URL = ''http://localhost:18080'' }
$apiKeyPath = ''C:\\Secure\\kabu_api_key.txt''
if (-not $env:KABU_API_KEY -and (Test-Path $apiKeyPath)) {
  $env:KABU_API_KEY = (Get-Content $apiKeyPath -Raw).Trim()
}

if ($DryRun) {
  $env:MODE = ''DRYRUN''
} elseif (-not $env:MODE) {
  $env:MODE = ''PAPER''
}

$pythonExe = ''py''
$venvPython = Join-Path -Path (Get-Location) -ChildPath ''.\.venv\Scripts\python.exe''
if (Test-Path $venvPython) {
  $pythonExe = (Resolve-Path $venvPython).Path
}

switch ($Action) {
  ''price'' {
    if (-not $Symbol) { throw ''-Symbol is required for price lookup'' }
    & $pythonExe -c "from exec.api_client import get_board; import json; print(json.dumps(get_board('$Symbol'), ensure_ascii=False, indent=2))"
    break
  }
  ''run'' {
    & $pythonExe orchestrate\run_intraday.py
    break
  }
  ''buy'' { $side = ''BUY'' }
  ''sell'' { $side = ''SELL'' }
}

if ($Action -in @(''buy'',''sell'')) {
  if (-not $Symbol) { throw ''-Symbol is required for trade actions'' }
  $argsList = @(''exec\kabu_exec.py'', ''--symbol'', $Symbol, ''--side'', $side, ''--qty'', $Qty)
  if ($PSBoundParameters.ContainsKey(''Price'')) {
    $argsList += @(''--entry'', $Price)
  }
  & $pythonExe @argsList
}

Write-Host ''Done.''
