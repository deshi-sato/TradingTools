param(
  [Parameter(Mandatory=$true)][string]$Date,  
  [string]$Symbols = "",                      
  [double]$Speed = 3.0,
  [switch]$Rebuild                            
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location $Root\..\

# --- 0. basic checks ---
if (-not (Test-Path .\scripts)) { throw "scripts folder not found" }
if (-not (Test-Path .\db\naut_market.db)) { throw "db\naut_market.db not found" }
if (-not (Test-Path .\scripts\__init__.py)) { ni .\scripts\__init__.py -Force | Out-Null }

# --- 1. rebuild features if needed ---
$srcDb = "db\naut_market.db"
if ($Rebuild) {
  Write-Host "Rebuilding features_stream from orderbook_snapshot..."
  py scripts\rebuild_features_from_orderbook.py
  $srcDb = "db\naut_market_refeed.db"
  if (-not (Test-Path $srcDb)) { throw "$srcDb not created" }
}

# --- 2. auto-detect symbols if not given ---
function Get-DateSymbols([string]$DbPath, [string]$DateStr) {
  $py = @"
import sqlite3
db = r"$DbPath"
d  = "$DateStr"
con = sqlite3.connect(db)
con.row_factory = sqlite3.Row
q = f"SELECT DISTINCT symbol FROM features_stream WHERE date(datetime(t_exec,'unixepoch','localtime'))=? ORDER BY symbol"
rows = [r['symbol'] for r in con.execute(q, (d,))]
print(','.join(rows))
"@
  $out = @($py) | py -
  return ($out -join "").Trim()
}

if ([string]::IsNullOrWhiteSpace($Symbols) -or $Symbols.Trim().ToUpper() -eq "AUTO") {
  $autoListCsv = Get-DateSymbols -DbPath $srcDb -DateStr $Date
  if ([string]::IsNullOrWhiteSpace($autoListCsv)) {
    throw "No symbols found for date $Date in features_stream."
  }
  $autoSyms = $autoListCsv -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }

  Write-Host ($Date + " symbols detected: " + ($autoSyms -join ", "))
  for ($i=0; $i -lt $autoSyms.Count; $i++) {
    ("{0,2}: {1}" -f ($i+1), $autoSyms[$i]) | Write-Host
  }
  $choice = Read-Host "Select symbols [ALL / 1,3,5 / 8136,5016 ...]"
  if ([string]::IsNullOrWhiteSpace($choice) -or $choice.Trim().ToUpper() -eq "ALL") {
    $Symbols = ($autoSyms -join ",")
  } else {
    $parts = $choice -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
    $sel = New-Object System.Collections.Generic.List[string]
    foreach ($p in $parts) {
      if ($p -match '^\d+$') {
        $idx = [int]$p - 1
        if ($idx -ge 0 -and $idx -lt $autoSyms.Count) { $sel.Add($autoSyms[$idx]) }
      } elseif ($autoSyms -contains $p) {
        $sel.Add($p)
      }
    }
    if ($sel.Count -eq 0) { throw "No valid selection." }
    $Symbols = ($sel.ToArray() -join ",")
  }
}

# --- 3. prepare replay DB ---
Copy-Item $srcDb db\naut_market_replay.db -Force

# --- 4. write config file ---
$cfg = @"
{
  "market_db": "db\\naut_market_replay.db",
  "ops_db": "db\\naut_ops.db",
  "trading_start": "00:00",
  "trading_end": "23:59",
  "runner_max_hold_sec": 20,
  "cooldown_sec": 5,
  "log_level": "INFO",
  "symbols": [$(($Symbols -split "," | ForEach-Object { '"' + $_.Trim() + '"' }) -join ", ")]
}
"@
$cfgPath = "config\stream_settings.replay.json"
$cfg | Set-Content -Encoding UTF8 $cfgPath

Write-Host ("Symbols used: " + $Symbols)
Write-Host ("Config written: " + $cfgPath)

# --- 5. run replay and runner inline (real-time output) ---
Write-Host "===== Running replay_naut.py ====="
py scripts\replay_naut.py -Src $srcDb -Dst db\naut_market_replay.db -Date $Date -Symbols $Symbols -NoSleep

Write-Host "`n===== Running naut_runner.py ====="
py -m scripts.naut_runner -Config $cfgPath

Write-Host "`n===== Replay finished ====="

# --- 6. generate summary ---
$py2 = @"
import sqlite3, os, pandas as pd
ops = r"db\\naut_ops.db"
date_str = "$Date"
outdir = r"data\\analysis"
os.makedirs(outdir, exist_ok=True)
con = sqlite3.connect(ops)

q = f'''
SELECT symbol,
       COUNT(*) AS trades,
       SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS winrate,
       ROUND(SUM(pnl),2)  AS sum_pnl,
       ROUND(AVG(pnl),2)  AS avg_pnl,
       ROUND(MAX(pnl),2)  AS max_pnl,
       ROUND(MIN(pnl),2)  AS min_pnl
FROM paper_pairs
WHERE date(datetime(exit_ts,'unixepoch','localtime')) = '{date_str}'
GROUP BY symbol
ORDER BY sum_pnl DESC
'''
df = pd.read_sql(q, con)
print(df)
df.to_csv(os.path.join(outdir, f"pnl_by_symbol_{date_str}.csv"), index=False, encoding="utf-8")
print("Saved to", os.path.join(outdir, f"pnl_by_symbol_{date_str}.csv"))
"@
$py2 | py -
