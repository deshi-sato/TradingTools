# cleanup_project.ps1
# プロジェクト直下で実行してください
# 使い方:
#   .\cleanup_project.ps1          # ドライラン（プレビューのみ）
#   .\cleanup_project.ps1 -DoIt    # 実行（実際に移動）

param(
  [switch]$DoIt
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# --- 目標ディレクトリを用意 ---
$dirs = @(
  "scripts",
  "data\db",
  "data\analysis",
  "logs",
  "history"
)
foreach ($d in $dirs) {
  if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

# --- 隔離フォルダ（削除はしない） ---
$ts = Get-Date -Format "yyyyMMdd"
$Stash = Join-Path $Root ("_stash_{0}" -f $ts)
if (-not (Test-Path $Stash)) { New-Item -ItemType Directory -Path $Stash | Out-Null }

# --- ホワイトリスト（必須ファイルはそのまま） ---
$keepExact = @(
  "run_report.ps1",
  "summarize_wf_results.py",
  "db_updater.py",
  "requirements.txt",
  "report.md"
)

# --- 直下にあるべき “最新成果物” パターン ---
$keepGlobs = @(
  "compare_*.csv",
  "compare_*.png"
)

# --- 既知の移動ルール ---
$moveRules = @(
  @{ Glob = "*.db";                 Dest = "data\db"       },  # DB は data\db へ
  @{ Glob = "wf_summary_*.*";       Dest = "data\analysis" },  # 互換用の元出力
  @{ Glob = "scripts\*.py";         Dest = "scripts"       },  # 念のため二重化しても同じ所へ
  @{ Glob = "*.ipynb";              Dest = "data\analysis" },  # ノート類は分析へ
  @{ Glob = "*.tmp";                Dest = $Stash          },
  @{ Glob = "*_tmp.*";              Dest = $Stash          },
  @{ Glob = "*_backup.*";           Dest = $Stash          },
  @{ Glob = "*.bak";                Dest = $Stash          }
)

# --- .venv, logs, history はそのまま ---
$preserveDirs = @("^\.venv($|\\)", "^logs($|\\)", "^history($|\\)", "^data($|\\)", "^scripts($|\\)")

function Test-PreservePath([string]$path) {
  $rel = Resolve-Path $path | ForEach-Object { $_.Path.Replace($Root + "\","") }
  foreach ($p in $preserveDirs) { if ($rel -match $p) { return $true } }
  return $false
}

# --- 対象候補（隠し/システム除外） ---
$all = Get-ChildItem -Recurse -Force -File |
  Where-Object { -not ($_.Attributes -band [IO.FileAttributes]::System) }

# --- 処理計画を作る ---
$plan = New-Object System.Collections.Generic.List[object]

foreach ($f in $all) {
  $rel = $f.FullName.Replace($Root + "\","")

  # 1) preserve ディレクトリ配下はスキップ（中の個別移動は moveRules が担当）
  if (Test-PreservePath $f.DirectoryName) {
    continue
  }

  # 2) ホワイトリスト（直下の必須）
  if ($keepExact -contains $rel) {
    continue
  }
  foreach ($g in $keepGlobs) {
    if ($rel -like $g) { continue 2 }
  }

  # 3) ルールに基づく移動先を決める
  $dest = $null
  foreach ($r in $moveRules) {
    if ($rel -like $r.Glob) { $dest = $r.Dest; break }
  }

  if (-not $dest) {
    # 既知でないファイルは Stash へ
    $dest = $Stash
  }

  $target = Join-Path $dest (Split-Path $rel -Leaf)
  $plan.Add([PSCustomObject]@{
    From = $rel
    To   = $target
  }) | Out-Null
}

# --- プレビュー表示 ---
Write-Host "==== Cleanup Plan ===="
if ($plan.Count -eq 0) {
  Write-Host "No files to move. Looks clean."
} else {
  $plan | Format-Table -AutoSize
}

if (-not $DoIt) {
  Write-Host "`n(ドライランです) 実行するには: .\cleanup_project.ps1 -DoIt"
  exit 0
}

# --- 実行フェーズ ---
foreach ($step in $plan) {
  $toDir = Split-Path -Parent $step.To
  if (-not (Test-Path $toDir)) { New-Item -ItemType Directory -Path $toDir | Out-Null }
  try {
    Move-Item -LiteralPath (Join-Path $Root $step.From) -Destination $step.To -Force
    Write-Host ("Moved: {0} -> {1}" -f $step.From, $step.To)
  } catch {
    Write-Warning ("Failed: {0} -> {1} : {2}" -f $step.From, $step.To, $_.Exception.Message)
  }
}

Write-Host "Cleanup done."
