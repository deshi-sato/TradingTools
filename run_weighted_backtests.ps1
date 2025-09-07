$ErrorActionPreference = 'Stop'

# Root of the project
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
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

$scenarios = @(
  @{ name="baseline_55_30_15"; trend=0.55; vol=0.30; momo=0.15 },
  @{ name="trend_70_20_10";    trend=0.70; vol=0.20; momo=0.10 },
  @{ name="volume_40_50_10";   trend=0.40; vol=0.50; momo=0.10 }
)

$summaryRows = @()
foreach ($s in $scenarios) {
  $tag = $s.name
  $log = Join-Path $logsDir ("bt_{0}.log" -f $tag)
  "[$(Get-Date -f 'yyyy-MM-dd HH:mm:ss')] $tag weights t=$($s.trend) v=$($s.vol) m=$($s.momo)" | Tee-Object -FilePath $log

  $picksPath = Join-Path $outDirPicks ("picks_{0}_{1}_{2}.csv" -f $tag,$Start,$End)
  & $Python ".\scripts\generate_picks_from_daily.py" `
    --db-path ".\rss_daily.db" `
    --start $Start --end $End `
    --w-trend $s.trend --w-volume $s.vol --w-momo $s.momo `
    --out $picksPath 2>&1 | Tee-Object -FilePath $log -Append

  if (-not (Test-Path $picksPath)) { Write-Warning "no picks: $picksPath"; continue }

  $metricsPath = Join-Path $outDirMetrics ("metrics_{0}_{1}_{2}.csv" -f $tag,$Start,$End)
  & $Python ".\scripts\eval_picks_open_close_sl.py" `
    --db-path ".\rss_daily.db" `
    --picks $picksPath `
    --from $Start --to $End `
    --out $metricsPath `
    --mode open_close `
    --sl 0.02 --tp 0.04 2>&1 | Tee-Object -FilePath $log -Append

  if (-not (Test-Path $metricsPath)) { Write-Warning "no metrics: $metricsPath"; continue }

  $m = Import-Csv $metricsPath | Select-Object -First 1
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

$summaryCsv = Join-Path $outSummary ("summary_{0}_{1}.csv" -f $Start,$End)
$summaryRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $summaryCsv
"Summary saved: $summaryCsv"

