cd C:\Users\Owner\documents\desshi_signal_viewer
C:/Users/Owner/Documents/desshi_signal_viewer/.venv/Scripts/Activate.ps1

    $FeatureDB = "c:\TradingToolsData\db\naut_market_20251111_refeed.db"
    $MLModel = ".\models\lstm_3905_v2.pt"
    $MLFeature = ".\exports\feature_names_3905.txt"

    py -m scripts.naut_runner `
    --symbols 3905 `
    --mode BUY `
    --broker paper `
    --ml-model $MLModel `
    --ml-feat-names $MLFeature `
    --features_db $FeatureDB `
    --prob-up-len 3 `
    --vol-ma3-thr 700 `
    --vol-rate-thr 1.30 `
    --sync-ticks 3 `
    --cooldown-ms 1500 `
    --verbose 1 `
    --feature-source raw_push `
    --window 09:00:00-15:30:00 `
    --replay-from-start
