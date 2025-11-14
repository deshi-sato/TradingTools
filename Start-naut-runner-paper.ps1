cd C:\Users\Owner\documents\TradingTools
C:/Users/Owner/Documents/TradingTools/.venv/Scripts/Activate.ps1

    $FeatureDB = "c:\TradingToolsData\db\naut_market_20251111_refeed.db"
    $MLModel = ".\models\lstm_3905_v2.pt"
    $MLFeature = ".\exports\feature_names_3905.txt"

    py -m scripts.naut_runner `
    --symbols 3905 `
    --mode BUY `
    --sentinel http://127.0.0.1:58900 `
    --prob-up-len 3 `
    --vol-ma3-thr 700 `
    --vol-rate-thr 1.30 `
    --spread-ticks 2 `
    --cooldown-sec 5 `
    --threshold-buy 0.8 `
    --threshold-sell 0.3