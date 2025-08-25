
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
    if (Test-Path $req) {
        Write-Host "[INFO] Installing requirements.txt ..."
        Invoke-Py -ArgList @("-m","pip","install","-r",$req)
    } else {
        Write-Host "requirements.txt not found. Skipping dependency install."
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
        Invoke-Py -ArgList @($yf)
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

        # 閾値（必要なら上で param に昇格させてください）
        $SharpeMin = 0.30

        # 読み込み → Calmar代理（cum_end / |max_dd|）→ 下限フィルタ → 降順ソート
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

        # 保存：ランキング
        $rankCsv = Join-Path $anaDir "ranked_strategies.csv"
        $ranked | Export-Csv $rankCsv -NoTypeInformation

        if ($ranked.Count -ge 1) {
            $buy  = ($ranked | Select-Object -First 1).tag
            $sell = ($ranked | Select-Object -Last 1).tag

            # 保存：当日の買い/売りピック（履歴として追記）
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
    # 銘柄抽出（pick_tickers.py）: daily_strategy_pick.csv の最新行に従い BUY/SELL を1銘柄ずつ選定
    # =========================
    try {
        $PyExe      = Join-Path $Root ".venv\Scripts\python.exe"
        $PickScript = Join-Path $Root "scripts\pick_tickers.py"
        $AnaDir     = Join-Path $Root "data\analysis"
        $PickList   = Join-Path $AnaDir "daily_strategy_pick.csv"
        $DbPath     = Join-Path $Root "rss_daily.db"

        if (-not (Test-Path $PickList)) { throw "daily_strategy_pick.csv not found: $PickList" }
        if (-not (Test-Path $PickScript)) { throw "pick_tickers.py not found: $PickScript" }
        if (-not (Test-Path $PyExe)) { throw "Python not found: $PyExe" }
        if (-not (Test-Path $DbPath)) { throw "DB not found: $DbPath" }

        Write-Host "[INFO] Picking tickers from $PickList ..."
        # しきい値は必要に応じて調整: --min-vol-ma 0（出来高フィルタ無効）、--min-days 0（自動= max(T,V)+1）

        # ① ウォッチリスト（各15・サイズ付き）
        & $PyExe $PickScript `
            --db $DbPath `
            --from-daily-pick $PickList `
            --top-long 15 --top-short 15 `
            --size-mode atr --capital 1000000 --risk-pct 0.005 --atr-window 14 --atr-mult 1.5 `
            --lot 100 --min-notional 100000 --max-notional 2000000 `
            --out (Join-Path $AnaDir ("watchlist_{0}.csv" -f (Get-Date -Format 'yyyy-MM-dd'))) `
            2>&1

        # ② 最小ピック（各1・サイズ付き：強制発注デフォルト）
        & $PyExe $PickScript `
            --db $DbPath `
            --from-daily-pick $PickList `
            --top-long 1 --top-short 1 `
            --size-mode atr --capital 1000000 --risk-pct 0.005 --atr-window 14 --atr-mult 1.5 `
            --lot 100 --min-notional 100000 --max-notional 2000000 `
            2>&1

        # 直近の picks_*.csv を表示（確認用）
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
    # 成果物の存在確認（data\analysis 内をチェック）
    # =========================
    $src = Join-Path $Root "data\analysis"
    $mustExist = @(
        "compare_summary.csv",
        "compare_monthly_table.csv",
        "compare_cum.png",
        "compare_monthly_mean.png"
    )
    $missing = @()
    foreach ($f in $mustExist) {
        if (-not (Test-Path (Join-Path $src $f))) { $missing += $f }
    }
    if ($missing.Count -gt 0) {
        Write-Warning ("Missing outputs:`n - " + ($missing -join "`n - "))
    } else {
        Write-Host "Done."
    }
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
