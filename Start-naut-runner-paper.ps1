cd C:\Users\Owner\documents\desshi_signal_viewer
C:/Users/Owner/Documents/desshi_signal_viewer/.venv/Scripts/Activate.ps1

    $ThresholdPath = 'exports\best_thresholds_buy_REF20251022_0901.json'
    $ConfigPath = 'config\runner_settings.json'
    $PolicyPath = 'config\policy.default.json'
    $LogPath = "logs\naut_runner_paper_20251029.log"
    $FeatureDB = "db\naut_market_20251029_refeed.db"
    $OpsDB = "db\naut_ops_20251029.db"

    py -m scripts.naut_runner `
    --symbols 1960 6330 290A `
    --mode BUY `
    --broker paper `
    --features_db $FeatureDB `
    --ops_db $OpsDB `
    --thr $ThresholdPath `
    --config $ConfigPath `
    --policy $PolicyPath `
    --log_path $LogPath `
    --verbose 1 `
    --replay-from-start

    --feature-source raw_push `
