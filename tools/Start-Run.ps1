param(
    [string]$Mode = "PAPER",
    [int]$Minutes = 20
)

# ========== ログ設定 ==========
$ts  = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = "logs/intraday-$ts.log"

Write-Output ">>> intraday starting at $ts, Mode=$Mode, Minutes=$Minutes"
Write-Output ">>> logging to $log"

# ========== 環境変数 ==========
$env:MODE      = $Mode
$env:RUN_MINUTES = $Minutes

# ========== intraday 実行 ==========
# 標準出力・エラーをまとめてログに保存
py -m orchestrate.run_intraday *> $log 2>&1
