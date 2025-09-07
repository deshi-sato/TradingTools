$ErrorActionPreference = 'Stop'

$Root  = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = "python"
$Start  = "2025-01-01"
$End    = (Get-Date).ToString('yyyy-MM-dd')

$outDirPicks   = Join-Path $Root "backtest\picks"
$outDirMetrics = Join-Path $Root "backtest\metrics"
$outSummary    = Join-Path $Root "backtest\summary"
$logsDir       = Join-Path $Root "logs"
foreach ($d in @($outDirPicks,$outDirMetrics,$outSummary,$logsDir)) {
  if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

# Additional intermediate scenarios
$scenarios = @(
  @{ name="trend_60_30_10"; trend=0.60; vol=0.30; momo=0.10 },
  @{ name="trend_50_40_10"; trend=0.50; vol=0.40; momo=0.10 }
)

# Load existing summary if present
$summaryCsv = Join-Path $outSummary ("summary_{0}_{1}.csv" -f $Start,$End)
$summaryRows = @()
if (Test-Path $summaryCsv) {
  $summaryRows = Import-Csv $summaryCsv
}

foreach ($s in $scenarios) {
  $tag = $s.name
  $log = Join-Path $logsDir ("bt_{0}.log" -f $tag)
  "[$(Get-Date -f 'yyyy-MM-dd HH:mm:ss')] $tag weights t=$($s.trend) v=$($s.vol) m=$($s.momo)" | Tee-Object -FilePath $log

  $picksPath = Join-Path $outDirPicks ("picks_{0}_{1}_{2}.csv" -f $tag,$Start,$End)
  if (-not (Test-Path $picksPath)) {
    & $Python ".\scripts\generate_picks_from_daily.py" `
      --db-path ".\rss_daily.db" `
      --start $Start --end $End `
      --w-trend $s.trend --w-volume $s.vol --w-momo $s.momo `
      --out $picksPath 2>&1 | Tee-Object -FilePath $log -Append
  } else {
    "exists: $picksPath" | Tee-Object -FilePath $log -Append
  }

  if (-not (Test-Path $picksPath)) { Write-Warning "no picks: $picksPath"; continue }

  $metricsPath = Join-Path $outDirMetrics ("metrics_{0}_{1}_{2}.csv" -f $tag,$Start,$End)
  if (-not (Test-Path $metricsPath)) {
    & $Python ".\scripts\eval_picks_open_close_sl.py" `
      --db-path ".\rss_daily.db" `
      --picks $picksPath `
      --from $Start --to $End `
      --out $metricsPath `
      --mode open_close `
      --sl 0.02 --tp 0.04 2>&1 | Tee-Object -FilePath $log -Append
  } else {
    "exists: $metricsPath" | Tee-Object -FilePath $log -Append
  }

  if (-not (Test-Path $metricsPath)) { Write-Warning "no metrics: $metricsPath"; continue }

  $m = Import-Csv $metricsPath | Select-Object -First 1

  # Remove existing row with same Scenario to avoid duplicates
  $summaryRows = $summaryRows | Where-Object { $_.Scenario -ne $tag }

  $summaryRows += [pscustomobject]@{
    Scenario    = $tag
    W_Trend     = $s.trend
    W_Volume    = $s.vol
    W_Momo      = $s.momo
    WinRate     = $m.WinRate
    AvgReturn   = $m.AvgReturn
    RewardRisk  = $m.RewardRisk
    Trades      = $m.Trades
    PicksFile   = $picksPath
    MetricsFile = $metricsPath
  }
}

$summaryRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $summaryCsv
"Summary updated: $summaryCsv"

