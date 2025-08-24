# Safe runner: uses helper Python scripts (no python -c quoting)
$ErrorActionPreference = "Stop"
function Die($msg){ Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

$DB     = "data\rss_daily.db"
$TABLE  = "daily_bars"
$OUTCSV = "data\score_daily.csv"
$RANK   = "spearman"

if (-not (Test-Path $DB)) { Die "DB not found: $DB" }
if (-not (Test-Path "scripts\score_tuner.py")) { Die "scripts\score_tuner.py not found" }
if (-not (Test-Path "scripts\analyze_scores.py")) { Die "scripts\analyze_scores.py not found" }
if (-not (Test-Path "scripts\get_recent_dates.py")) { Die "scripts\get_recent_dates.py not found" }

# 1) dates
$dates = python scripts\get_recent_dates.py
if (-not $dates) { Die "could not fetch dates from DB" }
$parts = $dates -split ","
if ($parts.Count -lt 2) { Die "need 2 dates, got: $dates" }
$START = $parts[0].Trim()
$END   = $parts[1].Trim()
Write-Host "Dates: $START -> $END"

# 2) score_tuner
Write-Host "`n[1/3] Running score_tuner.py..."
python scripts\score_tuner.py `
  --db $DB `
  --table $TABLE `
  --date_col date `
  --code_col ticker `
  --price_col close `
  --volume_col volume `
  --start $START `
  --end   $END `
  --rank $RANK `
  --out $OUTCSV

if (-not (Test-Path $OUTCSV)) { Die "score_tuner.py did not produce $OUTCSV" }

# 3) add date column if missing
$head = Get-Content -Head 1 $OUTCSV
if ($head -notmatch '^date,') {
  Write-Host "[patch] add date column to $OUTCSV ($END)"
  python scripts\add_date_to_score_csv.py
  $head = Get-Content -Head 1 $OUTCSV
  if ($head -notmatch '^date,') { Die "failed to add date column to $OUTCSV" }
}

# 4) analyze
Write-Host "`n[2/3] Running analyze_scores.py..."
python scripts\analyze_scores.py

# 5) results
Write-Host "`n[3/3] Results"
if (Test-Path "data\analysis\summary.txt") { Get-Content -Head 30 data\analysis\summary.txt }
if (Test-Path "data\analysis\ic_by_day.csv") { Write-Host "`nIC first rows:"; Get-Content -Head 5 data\analysis\ic_by_day.csv }
if (Test-Path "data\analysis\topn_by_day.csv") { Write-Host "`nTopN first rows:"; Get-Content -Head 5 data\analysis\topn_by_day.csv }
Write-Host "`nDone."
