cd C:\Users\Owner\documents\desshi_signal_viewer
C:/Users/Owner/Documents/desshi_signal_viewer/.venv/Scripts/Activate.ps1

param(
    [string[]]$Symbols = @('215A','3905','338A'),
    [string]$ThresholdPath = 'exports\best_thresholds_buy_REF20251022_0901.json',
    [string]$FeaturesDb = 'db\naut_market_refeed.db',
    [string]$OpsDb = 'db\naut_ops.db',
    [string]$ConfigPath = 'config\best_thresholds_buy_latest.json',
    [string]$PolicyPath = 'config\policy.default.json'
)

$runnerConfig = [ordered]@{
    features_db                  = $FeaturesDb
    ops_db                       = $OpsDb
    symbols                      = $Symbols
    poll_interval_sec            = 0.5
    initial_cash                 = 1500000
    fee_rate_bps                 = 2.0
    slippage_ticks               = 0.5
    tick_size                    = 0.1
    tick_value                   = 1.0
    min_lot                      = 100
    risk_per_trade_pct           = 0.01
    max_concurrent_positions     = 3
    daily_loss_limit_pct         = 0.04
    stats_interval_sec           = 300
    stop_loss_ticks              = 6
    log_path                     = 'logs\naut_runner_paper_20251017.log'
    killswitch_check_interval_sec = 5
}

$runnerConfig | ConvertTo-Json -Depth 4 | Set-Content -Path $ConfigPath -Encoding UTF8
Write-Host "Runner config written to $ConfigPath" -ForegroundColor Cyan

py -m scripts.naut_runner `
    --mode BUY `
    --broker paper `
    --thr $ThresholdPath `
    --config $ConfigPath `
    --policy $PolicyPath `
    --verbose 1 `
    --replay-from-start
