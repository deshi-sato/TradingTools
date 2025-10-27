cd C:\Users\Owner\documents\desshi_signal_viewer
C:/Users/Owner/Documents/desshi_signal_viewer/.venv/Scripts/Activate.ps1

    $ThresholdPath = 'exports\best_thresholds_buy_REF20251022_0901.json'
    $ConfigPath = 'config\best_thresholds_buy_latest.json'
    $PolicyPath = 'config\policy.default.json'

    py -m scripts.naut_runner `
    --mode BUY `
    --broker paper `
    --thr $ThresholdPath `
    --config $ConfigPath `
    --policy $PolicyPath `
    --verbose 0 `
    --replay-from-start
