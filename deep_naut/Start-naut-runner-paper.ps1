cd C:\Users\Owner\documents\TradingTools\deep_naut
.venv/Scripts/Activate.ps1

    $FeatureDB = ".\db\naut_market_20251107_refeed.db"
    $MLModel = ".\models\lstm_3905_v2.pt"
    $MLFeature = ".\exports\feature_names_3905.txt"

    py -m scripts.naut_runner `
    --symbols 3905 `
    --mode BUY `
    --broker paper --dry-run 1 `
    --features_db $FeatureDB `
    --ml-model $MLModel `
    --ml-feat-names $MLFeature `
    --prob-up-len 3 `
    --vol-ma3-thr 700 `
    --vol-rate-thr 1.30 `
    --vol-gate OR `
    --sync-ticks 3 `
    --cooldown-ms 1500 `
    --verbose 1 `
    --feature-source raw_push `
    --replay-from-start
