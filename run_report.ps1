param(
  [string]$Script = "scripts\summarize_wf_results.py",
  [string]$Start  = "2025-01-01",
  [string]$End    = (Get-Date).ToString('yyyy-MM-dd'),
  [switch]$DryRun,
  [switch]$SkipPipUpgrade
)

function Get-InvalidCharRegex {
  $bad = [IO.Path]::GetInvalidFileNameChars()
  return ('[{0}]' -f ([regex]::Escape(($bad -join ''))))
}

function Normalize-PathComponent {
  param([Parameter(Mandatory)][string]$Text)
  $re = Get-InvalidCharRegex
  return ($Text -replace $re, '_')
}

function Test-PathVerbose {
  param(
    [Parameter(Mandatory)][string]$Path,
    [string]$Label = "path"
  )
  $reFile = Get-InvalidCharRegex
  $rePath = ('[{0}]' -f ([regex]::Escape(([IO.Path]::GetInvalidPathChars() -join ''))))
  Write-Host "[DEBUG] Test-Path <$Label>: $Path"
  $check = $Path
  if ($check -match '^[A-Za-z]:') { $check = $check.Substring(2) }
  if ($check -match $rePath) {
    $bad = $Matches[0]
    throw "Invalid char '$bad' in <$Label>: $Path"
  }
  $parts = $check -split '[\\/]'
  foreach ($part in $parts) {
    if ([string]::IsNullOrEmpty($part)) { continue }
    if ($part -match $reFile) {
      $bad = $Matches[0]
      throw "Invalid char '$bad' in <$Label>: $Path"
    }
  }
  return (Test-Path -LiteralPath $Path)
}

function Test-FileLocked {
  param([Parameter(Mandatory)][string]$Path)
  try {
    $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    $fs.Close()
    return $false
  } catch {
    return $true
  }
}

$ErrorActionPreference = 'Stop'
$Root    = $PSScriptRoot
$logsDir = Join-Path $Root 'logs'
if (-not (Test-Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }
$stamp   = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = Join-Path $logsDir "task_run_$stamp.log"

Start-Transcript -Path $logFile -Append
"== Started: $stamp  User=$env:USERNAME  WD=$(Get-Location) ==" | Out-Host

try {
    try {
        chcp 65001 | Out-Null
        [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
        $OutputEncoding = [System.Text.Encoding]::UTF8
    } catch {}

    Set-Location $Root
    Write-Host "[INFO] Project root: $Root"

    $PyExe = Join-Path $Root ".venv\Scripts\python.exe"
    if (-not (Test-PathVerbose -Path $PyExe -Label "venv python")) {
        Write-Host "[INFO] Creating venv at .venv ..."
        if (Get-Command py -ErrorAction SilentlyContinue) { py -3 -m venv .venv }
        else { python -m venv .venv }
        if (-not (Test-PathVerbose -Path $PyExe -Label "venv python")) { throw "venv creation failed: $PyExe not found." }
    }

    function Invoke-Py {
        param([string[]]$ArgList)
        & $PyExe @ArgList
        $code = $LASTEXITCODE
        if ($code -ne 0) { throw "Python exited with code $code." }
    }

    if (-not $SkipPipUpgrade) {
        $env:PIP_DEFAULT_TIMEOUT = "60"
        Write-Host "Upgrading pip/setuptools/wheel ..."
        Invoke-Py -ArgList @(
          "-m","pip","install","--upgrade","pip","setuptools","wheel",
          "--no-input","--disable-pip-version-check","--timeout","60","--retries","2"
        )
    } else {
        Write-Host "[INFO] Skip pip upgrade."
    }

    $req = Join-Path $Root "requirements.txt"
    if ((Test-PathVerbose -Path $req -Label "requirements") -and ((Get-Content $req -ErrorAction SilentlyContinue) -join "" -ne "")) {
        Write-Host "[INFO] Installing requirements.txt ..."
        Invoke-Py -ArgList @("-m","pip","install","-r",$req)
    } else {
        Write-Host "requirements.txt not found or empty. Skipping dependency install."
    }

    if ($DryRun) {
        Write-Host "[DRY-RUN] Stopping before executing Python jobs."
        return
    }

    $yf = Join-Path $Root "scripts\fetch_yf_daily.py"
    if (Test-PathVerbose -Path $yf -Label "fetch_yf_daily.py") {
        Write-Host "[INFO] Updating rss_daily.db via scripts\fetch_yf_daily.py ..."
        Invoke-Py -ArgList @($yf, "--db", (Join-Path $Root "data\rss_daily.db"))
    } else {
        Write-Host "[INFO] scripts\fetch_yf_daily.py not found. Skipping daily DB update."
    }

    $pyScriptOrig = Join-Path $Root $Script
    $scriptDir    = Split-Path -Parent $Script
    $scriptLeaf   = Split-Path -Leaf   $Script
    $safeLeaf     = Normalize-PathComponent $scriptLeaf
    $relScript    = if ([string]::IsNullOrEmpty($scriptDir)) { $safeLeaf } else { Join-Path $scriptDir $safeLeaf }
    $pyScriptSafe = Join-Path $Root $relScript

    if (Test-PathVerbose -Path $pyScriptOrig -Label "summary script (orig)") {
        $pyScript = $pyScriptOrig
    } elseif (Test-PathVerbose -Path $pyScriptSafe -Label "summary script (sanitized)") {
        $pyScript = $pyScriptSafe
    } else {
        throw "Python script not found: $pyScriptSafe"
    }

    Write-Host "Running: $pyScript --start $Start --end $End"
    Invoke-Py -ArgList @($pyScript,"--start",$Start,"--end",$End)

    try {
        $anaDir = Join-Path $Root "data\analysis"
        $sumCsv = Join-Path $anaDir "compare_summary.csv"
        if (-not (Test-PathVerbose -Path $sumCsv -Label "compare_summary.csv")) { throw "compare_summary.csv not found: $sumCsv" }

        $SharpeMin = 0.30

        $ranked = Import-Csv $sumCsv | ForEach-Object {
            $tag = $_.tag
            $sh  = [double]($_.sharpe)
            $cr  = [double]($_.cum_end)
            $dd  = [double]($_.max_dd)
            [pscustomobject]@{
                tag          = $tag
                sharpe       = $sh
                cum_return   = $cr
                max_dd       = $dd
                calmar_proxy = if ([math]::Abs($dd) -gt 1e-9) { $cr / [math]::Abs($dd) } else { [double]::NaN }
            }
        } | Where-Object { $_.sharpe -ge $SharpeMin -and $_.cum_return -gt 0 -and $_.max_dd -lt 0 } |
            Sort-Object calmar_proxy -Descending

        $rankCsv = Join-Path $anaDir "ranked_strategies.csv"
        $ranked | Export-Csv $rankCsv -NoTypeInformation

        if ($ranked.Count -ge 1) {
            $buy  = ($ranked | Select-Object -First 1).tag
            $sell = ($ranked | Select-Object -Last 1).tag

            $pickCsv = Join-Path $anaDir "daily_strategy_pick.csv"
            $row = [pscustomobject]@{ date = (Get-Date -Format 'yyyy-MM-dd'); buy = $buy; sell = $sell }
            if (Test-PathVerbose -Path $pickCsv -Label "daily_strategy_pick.csv (existing)") { $row | Export-Csv $pickCsv -NoTypeInformation -Append }
            else { $row | Export-Csv $pickCsv -NoTypeInformation }

            Write-Host "[INFO] Ranked $(($ranked).Count) strategies. BUY=$buy  SELL=$sell"
        }
        else {
            Write-Warning "No strategies passed filters (SharpeMin=$SharpeMin). ranked_strategies.csv is empty."
        }
    }
    catch {
        Write-Warning "Ranking step skipped: $($_.Exception.Message)"
    }

    try {
        $PyExe      = Join-Path $Root ".venv\Scripts\python.exe"
        $PickScript = Join-Path $Root "scripts\pick_tickers.py"
        $AnaDir     = Join-Path $Root "data\analysis"
        $PickList   = Join-Path $AnaDir "daily_strategy_pick.csv"
        $DbPath     = Join-Path $Root "data\rss_daily.db"

        if (-not (Test-PathVerbose -Path $PickList -Label "daily_strategy_pick.csv")) { throw "daily_strategy_pick.csv not found: $PickList" }
        if (-not (Test-PathVerbose -Path $PickScript -Label "pick_tickers.py")) { throw "pick_tickers.py not found: $PickScript" }
        if (-not (Test-PathVerbose -Path $PyExe -Label "venv python")) { throw "Python not found: $PyExe" }
        if (-not (Test-PathVerbose -Path $DbPath -Label "rss_daily.db")) { throw "DB not found: $DbPath" }

        Write-Host "[INFO] Picking tickers from $PickList ..."

        $oldEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $PyExe $PickScript `
            --db $DbPath `
            --from-daily-pick $PickList `
            --top-long 15 --top-short 15 `
            --size-mode atr --capital 1000000 --risk-pct 0.005 --atr-window 14 --atr-mult 1.5 `
            --lot 100 --min-notional 100000 --max-notional 2000000 `
            --out (Join-Path $AnaDir ("watchlist_{0}.csv" -f (Get-Date -Format 'yyyy-MM-dd')))
        $ErrorActionPreference = $oldEap

        $oldEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $PyExe $PickScript `
            --db $DbPath `
            --from-daily-pick $PickList `
            --top-long 1 --top-short 1 `
            --size-mode atr --capital 1000000 --risk-pct 0.005 --atr-window 14 --atr-mult 1.5 `
            --lot 100 --min-notional 100000 --max-notional 2000000
        $ErrorActionPreference = $oldEap

        $lastPick = Get-ChildItem -Path $AnaDir -Filter "picks_*.csv" -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Desc | Select-Object -First 1
        if ($null -ne $lastPick) {
            Write-Host ("[INFO] Latest picks: {0} ({1})" -f $lastPick.Name, $lastPick.LastWriteTime)
        } else {
            Write-Warning "No picks_*.csv found under $AnaDir"
        }
    }
    catch {
        Write-Warning "Pick step skipped: $($_.Exception.Message)"
    }

    try {
        $Upd = Join-Path $Root "scripts\update_rss_snapshot_index.py"
        if (-not (Test-PathVerbose -Path $Upd -Label "update_rss_snapshot_index.py")) { throw "update_rss_snapshot_index.py not found: $Upd" }

        Write-Host "[INFO] Updating Excel index sheet ..."
        Invoke-Py -ArgList @(
            $Upd,
            "--data-dir", (Join-Path $Root "data\analysis"),
            "--excel",    (Join-Path $Root "stock_data.xlsm"),
            "--sheet",    "index",
            "--max-buy",  "15",
            "--max-sell", "15",
            "--log",      (Join-Path $Root "data\analysis\update_rss_snapshot_index.log")
        )
    }
    catch {
        Write-Warning "Excel index update skipped: $($_.Exception.Message)"
    }

    try {
        $Upd = Join-Path $Root "scripts\update_rss_snapshot_index.py"
        if (-not (Test-PathVerbose -Path $Upd -Label "update_rss_snapshot_index.py")) { throw "update_rss_snapshot_index.py not found: $Upd" }

        Write-Host "[INFO] Updating Excel index sheet (fallback) ..."
        Invoke-Py -ArgList @(
            $Upd,
            "--data-dir", (Join-Path $Root "data\analysis"),
            "--excel",    (Join-Path $Root "stock_data.xlsm"),
            "--sheet",    "index",
            "--max-buy",  "15",
            "--max-sell", "15",
            "--log",      (Join-Path $Root "data\analysis\update_rss_snapshot_index.log")
        )
    }
    catch {
        Write-Warning "Excel index update (fallback) skipped: $($_.Exception.Message)"
    }

    try {
        $Upd = Join-Path $Root "scripts\\update_rss_snapshot_index.py"
        if (Test-PathVerbose -Path $Upd -Label "update_rss_snapshot_index.py") {
            $excel = Join-Path $Root "stock_data.xlsm"
            if (Test-PathVerbose -Path $excel -Label "excel workbook") {
                $waited = 0
                while ((Test-FileLocked -Path $excel) -and ($waited -lt 30)) {
                    Write-Warning "Excel workbook locked by another process. Waiting 5s ..."
                    Start-Sleep -Seconds 5
                    $waited += 5
                }
                if (-not (Test-FileLocked -Path $excel)) {
                    try {
                        Write-Host "[INFO] Updating Excel index sheet (final retry) ..."
                        Invoke-Py -ArgList @(
                            $Upd,
                            "--data-dir", (Join-Path $Root "data\\analysis"),
                            "--excel",    $excel,
                            "--sheet",    "index",
                            "--max-buy",  "15",
                            "--max-sell", "15",
                            "--log",      (Join-Path $Root "data\\analysis\\update_rss_snapshot_index.log")
                        )
                    } catch {
                        Write-Warning "Excel update final retry failed: $($_.Exception.Message)"
                    }
                } else {
                    Write-Warning "Excel workbook still locked. Skipping final retry."
                }
            }
        }
    } catch {}

    $src = Join-Path $Root "data\analysis"
    $mustExist = @("compare_summary.csv","compare_monthly_table.csv","compare_cum.png","compare_monthly_mean.png")
    $missing = @()
    foreach ($f in $mustExist) {
        $p = Join-Path $src $f
        if (-not (Test-PathVerbose -Path $p -Label ("must-exist:{0}" -f $f))) { $missing += $f }
    }
    if ($missing.Count -gt 0) {
        Write-Warning ("Missing expected outputs: {0}" -f ($missing -join ", "))
    }

    Write-Host "Done."
}
catch {
    Write-Host "[ERROR] $($_.Exception.Message)"
    Write-Error $_
    exit 1
}
finally {
    "== Finished: $(Get-Date -Format 'yyyyMMdd_HHmmss') ==" | Out-Host
    Stop-Transcript
}

exit 0

