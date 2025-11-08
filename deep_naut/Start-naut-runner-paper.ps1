cd C:\Users\Owner\documents\desshi_signal_viewer
C:/Users/Owner/Documents/desshi_signal_viewer/.venv/Scripts/Activate.ps1

    $ThresholdPath = 'exports\best_thresholds_buy_REF20251022_0901.json'
    $ConfigPath = 'config\runner_settings.json'
    $PolicyPath = 'config\policy.default.json'
    $LogPath = "logs\naut_runner_paper_20251107.log"
    $FeatureDB = "db\naut_market_20251107_refeed.db"
    $OpsDB = "db\naut_ops_20251107.db"

    py -m scripts.naut_runner `
    --symbols 3905 `
    --mode BUY `
    --broker paper `
    --features_db $FeatureDB `
    --ops_db $OpsDB `
    --thr $ThresholdPath `
    --config $ConfigPath `
    --policy $PolicyPath `
    --log_path $LogPath `
    --verbose 1 `
    --feature-source raw_push `
    --replay-from-start
