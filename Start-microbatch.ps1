# Start-microbatch.ps1 stream_microbatch for smoketest
py -m scripts.kabus_login_wait -Config config\stream_settings.json
py -m scripts.build_fallback_scraper --no-browser
py -m scripts.build_watchlist
py -m scripts.register_watchlist -Config config\stream_settings.json -Input data\watchlist_today.csv -Max 3
$env:NAUT_SKIP_WAL = '1'
py -m scripts.stream_microbatch -Config config\stream_settings.json