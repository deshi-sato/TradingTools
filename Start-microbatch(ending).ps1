# Start-microbatch.ps1 stream_microbatch for smoketest(取引終了後作業）
$ds = @'
import glob,sqlite3
db = sorted(glob.glob(r"db\\naut_market_*_refeed.db"))[-1]
con = sqlite3.connect(db)
print(con.execute("SELECT dataset_id FROM dataset_registry ORDER BY created_at DESC LIMIT 1").fetchone()[0])
con.close()
'@ | py -
$ds
# リプレイ（イベント記録）
py -m scripts.replay_naut -Config config\stream_settings.json -DatasetId $ds -RunId RUN_$($ds) -Verbose 0

# ラベル生成（h=60,120s / +8,-6bp）
py -m scripts.build_labels_from_replay -DatasetId $ds -Horizons 60,120 -Thresholds +8,-6

# 学習セット出力
py -m scripts.build_training_set -DatasetId $ds

# BUY
py -m scripts.grid_search_thresholds -DatasetId $ds -MinTrades 50 -EVFloor -50 -CV 0 -Verbose 1
Get-Content ("exports\best_thresholds_{0}.json" -f $ds) | Write-Host

# SELL
py -m scripts.grid_search_thresholds_sell -DatasetId $ds -MinTrades 50 -EVFloor -50 -CV 0 -Verbose 1
Get-Content ("exports\best_thresholds_sell_{0}.json" -f $ds) | Write-Host
