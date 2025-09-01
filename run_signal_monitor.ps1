# ================== run_signal_monitor.ps1 ==================
# 監視: rss_snapshot.db の当日9:00以降データでシグナル検知
# 終了: BUYとSELLが揃う or 合計N件に到達 or デッドライン(=09:00+MaxMinutes)到達
# ログ/進捗はコンソールとファイルに出力

# ====== 設定 ======
$BaseDir          = "C:\Users\Owner\Documents\desshi_signal_viewer"
$PyExe            = "py"
$SignalPy         = Join-Path $BaseDir "signal_watcher.py"
$SnapshotDb       = Join-Path $BaseDir "data\rss_snapshot.db"
$IndexDb          = Join-Path $BaseDir "data\rss_index.db"   # 無い日は自動で --snapshot-only にします
$OutCsv           = Join-Path $BaseDir "out\signals_latest.csv"
$LogPath          = Join-Path $BaseDir "logs\signal_monitor.log"

# 監視頻度・終了条件
$PollSeconds      = 15                 # 監視間隔(秒)
$MaxMinutes       = 60                 # デッドライン(分) … 09:00 + 60 = 10:00
$StartHHmm        = "09:00"            # 当日の監視開始基準時刻(可変OK)

# 終了条件（「全部表示」で運用するなら下2行を緩めてもOK）
$RequireBothSides = $true              # BUYとSELLの両方が揃ったら終了
$MaxSignalsTotal  = 999999             # 合計N件に達したら終了（実質無効）

# ====== 初期化 ======
New-Item -ItemType Directory -Force -Path (Split-Path $OutCsv) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] === monitor start ===" | Out-File -Encoding utf8 -FilePath $LogPath

# 共有状態（scriptスコープで保持）
$script:Seen       = [System.Collections.Generic.HashSet[string]]::new()
$script:Confirmed  = [System.Collections.Generic.List[object]]::new()

# ====== 9:00起点でデッドラインを決定 ======
$today = Get-Date
$hh, $mm = $StartHHmm.Split(":",2)
$script:MarketOpen = Get-Date -Year $today.Year -Month $today.Month -Day $today.Day -Hour $hh -Minute $mm -Second 0
$script:Deadline   = $script:MarketOpen.AddMinutes($MaxMinutes)
Write-Host ("[init] window {0} → {1}" -f ($script:MarketOpen.ToString('HH:mm:ss')), ($script:Deadline.ToString('HH:mm:ss'))) -ForegroundColor Gray

# ====== ヘルパー ======
function Read-CsvSafe([string]$Path){
  if (-not (Test-Path $Path)) { return @() }
  try { return Import-Csv -Path $Path -Encoding UTF8 } catch {
    Start-Sleep -Milliseconds 200
    try { return Import-Csv -Path $Path -Encoding UTF8 } catch { return @() }
  }
}

function Invoke-SignalWatcher {
  $args = @(
    "--snapshot-db", $SnapshotDb,
    "--out", $OutCsv,
    "--log", $LogPath,
    "--start-hhmm", $StartHHmm,
    "--deadline-min", $MaxMinutes
  )
  if ($IndexDb -and (Test-Path $IndexDb)) {
    $args += @("--index-db", $IndexDb)
  } else {
    $args += @("--snapshot-only")
  }
  & $PyExe $SignalPy @args | Out-Null
}

function Capture-NewSignals {
  $rows = Read-CsvSafe -Path $OutCsv
  if (-not $rows) { return }

  foreach ($r in $rows) {
    $key = "{0}|{1}|{2}|{3}" -f $r.time,$r.ticker,$r.side,$r.strategy
    if ($script:Seen.Contains($key)) { continue }

    $script:Confirmed.Add($r)
    [void]$script:Seen.Add($key)

    "[$(Get-Date -Format 'HH:mm:ss')] NEW: $($r.side) $($r.ticker) $($r.strategy) entry=$($r.entry) stop=$($r.stop)" |
      Out-File -Append -Encoding utf8 -FilePath $LogPath
  }
}

function Should-Exit {
  if ($script:Confirmed.Count -ge $MaxSignalsTotal) { return $true }
  if ($RequireBothSides) {
    $hasBuy  = $script:Confirmed | Where-Object { $_.side -eq 'BUY' }  | Select-Object -First 1
    $hasSell = $script:Confirmed | Where-Object { $_.side -eq 'SELL' } | Select-Object -First 1
    if ($hasBuy -and $hasSell) { return $true }
  }
  return $false
}

# ====== 監視ループ ======
while ($true) {
  Invoke-SignalWatcher
  Capture-NewSignals

  $now = Get-Date
  Write-Host ("[{0}] monitoring...  confirmed={1}, deadline={2}" -f ($now.ToString('HH:mm:ss')), $script:Confirmed.Count, ($script:Deadline.ToString('MM/dd/yyyy HH:mm:ss'))) -ForegroundColor Cyan

  if (Should-Exit) {
    Write-Host ("[{0}] exit: goal reached. ({1} signals)" -f ($now.ToString('HH:mm:ss')), $script:Confirmed.Count) -ForegroundColor Green
    break
  }

  if ($now -ge $script:Deadline) {
    Write-Host ("[{0}] timeout reached (deadline)." -f ($now.ToString('HH:mm:ss'))) -ForegroundColor Yellow
    break
  }

  Start-Sleep -Seconds $PollSeconds
}

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] === monitor end ===" | Out-File -Append -Encoding utf8 -FilePath $LogPath
# ================== end of file ==================
