param(
  [string]$Script = "scripts\summarize_wf_results.py",
  [string]$Start  = "2025-01-01",
  [string]$End    = (Get-Date).ToString('yyyy-MM-dd'),
  [switch]$DryRun,
  [switch]$SkipPipUpgrade
)

# =========================
# Logging / bootstrap
# =========================
$ErrorActionPreference = 'Stop'
$Root    = $PSScriptRoot
$logsDir = Join-Path $Root 'logs'
if (-not (Test-Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }
$stamp   = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = Join-Path $logsDir "task_run_$stamp.log"

Start-Transcript -Path $logFile -Append
"== Started: $stamp  User=$env:USERNAME  WD=$(Get-Location) ==" | Out-Host

try {
    # --- Console を UTF-8（化け対策） ---
    try {
        chcp 65001 | Out-Null
        [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
        $OutputEncoding = [System.Text.Encoding]::UTF8
    } catch {}

    Set-Location $Root
    Write-Host "[INFO] Project root: $Root"

    # =========================
    # venv 準備
    # =========================
    $PyExe = Join-Path $Root ".venv\Scripts\python.exe"
    if (-not (Test-Path $PyExe)) {
        Write-Host "[INFO] Creating venv at .venv ..."
        if (Get-Command py -ErrorAction SilentlyContinue) { py -3 -m venv .venv }
        else { python -m venv .venv }
        if (-not (Test-Path $PyExe)) { throw "venv creation failed: $PyExe not found." }
    }

    function Invoke-Py {
        param([string[]]$ArgList)
        & $PyExe @ArgList
        $code = $LASTEXITCODE
        if ($code -ne 0) { throw "Python exited with code $code." }
    }

    # =========================
    # 依存の最小アップグレード & requirements
    # =========================
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
    if ((Test-Path $req) -and ((Get-Content $req -ErrorAction SilentlyContinue) -join "" -ne "")) {
        Write-Host "[INFO] Installing requirements.txt ..."
        Invoke-Py -ArgList @("-m","pip","install","-r",$req)
    } else {
        Write-Host "requirements.txt not found or empty. Skipping dependency install."
    }

    if ($DryRun) {
        Write-Host "[DRY-RUN] Stopping before executing Python jobs."
        return
    }

    # =========================
    # 日足DBの更新（あれば実行）
    # =========================
    $yf = Join-Path $Root "scripts\fetch_yf_daily.py"
    if (Test-Path $yf) {
        Write-Host "[INFO] Updating rss_daily.db via scripts\fetch_yf_daily.py ..."
        Invoke-Py -ArgList @($yf, "--db", (Join-Path $Root "data\rss_daily.db"))
    } else {
        Write-Host "[INFO] scripts\fetch_yf_daily.py not found. Skipping daily DB update."
    }

    # =========================
    # 集計レポート生成
    # =========================
    $pyScript = Join-Path $Root $Script
    if (-not (Test-Path $pyScript)) { throw "Python script not found: $pyScript" }

    Write-Host "Running: $pyScript --start $Start --end $End"
    Invoke-Py -ArgList @($pyScript,"--start",$Start,"--end",$End)

    # =========================
    # 戦略ランキング（Calmar代理＋Sharpe下限）と当日ピック
    # =========================
    try {
        $anaDir = Join-Path $Root "data\analysis"
        $sumCsv = Join-Path $anaDir "compare_summary.csv"
        if (-not (Test-Path $sumCsv)) { throw "compare_summary.csv not found: $sumCsv" }

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
            if (Test-Path $pickCsv) { $row | Export-Csv $pickCsv -NoTypeInformation -Append }
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

    # =========================
    # 銘柄抽出（pick_tickers.py）
    # =========================
    try {
        $PyExe      = Join-Path $Root ".venv\Scripts\python.exe"
        $PickScript = Join-Path $Root "scripts\pick_tickers.py"
        $AnaDir     = Join-Path $Root "data\analysis"
        $PickList   = Join-Path $AnaDir "daily_strategy_pick.csv"
        $DbPath     = Join-Path $Root "data\rss_daily.db"   # ← 修正: data\ 配下

        if (-not (Test-Path $PickList)) { throw "daily_strategy_pick.csv not found: $PickList" }
        if (-not (Test-Path $PickScript)) { throw "pick_tickers.py not found: $PickScript" }
        if (-not (Test-Path $PyExe)) { throw "Python not found: $PyExe" }
        if (-not (Test-Path $DbPath)) { throw "DB not found: $DbPath" }

        Write-Host "[INFO] Picking tickers from $PickList ..."

        & $PyExe $PickScript `
            --db $DbPath `
            --from-daily-pick $PickList `
            --top-long 15 --top-short 15 `
            --size-mode atr --capital 1000000 --risk-pct 0.005 --atr-window 14 --atr-mult 1.5 `
            --lot 100 --min-notional 100000 --max-notional 2000000 `
            --out (Join-Path $AnaDir ("watchlist_{0}.csv" -f (Get-Date -Format 'yyyy-MM-dd'))) `
            2>&1

        & $PyExe $PickScript `
            --db $DbPath `
            --from-daily-pick $PickList `
            --top-long 1 --top-short 1 `
            --size-mode atr --capital 1000000 --risk-pct 0.005 --atr-window 14 --atr-mult 1.5 `
            --lot 100 --min-notional 100000 --max-notional 2000000 `
            2>&1

        $lastPick = Get-ChildItem (Join-Path $AnaDir "picks_*.csv") -ErrorAction SilentlyContinue |
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

    # =========================
    # Watchlist → index 反映（Excel）
    # =========================
    try {
        $Upd = Join-Path $Root "scripts\update_rss_snapshot_index.py"
        if (-not (Test-Path $Upd)) { throw "update_rss_snapshot_index.py not found: $Upd" }

        Write-Host "[INFO] Updating Excel index sheet ..."
        Invoke-Py -ArgList @(
            $Upd,
            "--data-dir", (Join-Path $Root "data\analysis"),
            "--excel",    (Join-Path $Root "株価データ.xlsm"),
            "--sheet",    "index",
            "--max-buy",  "15",
            "--max-sell", "15",
            "--log",      (Join-Path $Root "data\analysis\update_rss_snapshot_index.log")
        )
    }
    catch {
        Write-Warning "Excel index update skipped: $($_.Exception.Message)"
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
