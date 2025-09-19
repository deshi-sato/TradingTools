param([int]$Limit = 70, [int]$FreshHours = 48)
$env:WATCHLIST_LIMIT   = "$Limit"
$env:RANKING_FRESH_HOURS = "$FreshHours"
py .\scripts\build_watchlist.py
Get-Content .\data\watchlist_today.csv | Select-Object -First 8
