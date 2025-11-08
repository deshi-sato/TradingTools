cd C:\Users\Owner\documents\desshi_signal_viewer
C:/Users/Owner/Documents/desshi_signal_viewer/.venv/Scripts/Activate.ps1

# Start-microbatch.ps1 stream_microbatch for smoketest
py -m scripts.kabus_login_wait -Config config\stream_settings.json
py -m scripts.build_fallback_scraper --no-browser
py -m scripts.build_watchlist
py -m scripts.register_watchlist -Config config\stream_settings.json -Input data\watchlist_today.csv -Max 1
$env:NAUT_SKIP_WAL = '1'
py -m scripts.stream_microbatch -Config config\stream_settings.json



#以下はグリッドサーチ起動用
# Start-microbatch.ps1 stream_microbatch for gridsearch
py scripts\resolve_gridsearch_target.py | Invoke-Expression

# スキーマ補正（再実行可）
py scripts\ensure_registry_schema.py $refeed

# ラベル作成（唯一の -DB 指定）
py -m scripts.build_labels_from_replay -DatasetId $ds -DB $refeed -Horizons 20 -Thresholds +8,-6

# トレーニングセット生成
py -m scripts.build_training_set -DatasetId $ds -Out ("exports\trainset_{0}.csv" -f $ds)

# グリッドサーチ
py -m scripts.grid_search_thresholds_buy -DatasetId $ds -Horizons 20 -MinTrades 50 -EVFloor 0 -CV 0
