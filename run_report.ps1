# ===== run_report.ps1 =====
#Requires -Version 5.1
param(
    # Pythonスクリプトの相対/絶対パス（省略時はルート直下の既定名）
    [string]$Script = "summarize_wf_results.py"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- Console をUTF-8（化け対策・絵文字は使わない） ---
try {
    chcp 65001 | Out-Null
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {}

# --- Paths ---------------------------------------------------------------
$Self  = $MyInvocation.MyCommand.Path
$Root  = Split-Path -Parent $Self
$Venv  = Join-Path $Root ".venv"
$PyExe = Join-Path $Venv "Scripts\python.exe"
$Req   = Join-Path $Root "requirements.txt"
$Logs  = Join-Path $Root "logs"

# Script は相対パスでもOK（scripts\xxx.py など）
if (-not (Test-Path $Script)) {
    $PyScript = Join-Path $Root $Script
} else {
    $PyScript = (Resolve-Path $Script).Path
}

# --- Ensure folders ------------------------------------------------------
if (-not (Test-Path $Logs)) { New-Item -ItemType Directory -Path $Logs | Out-Null }

# --- Create venv if missing ---------------------------------------------
if (-not (Test-Path $PyExe)) {
    Write-Host "Creating venv at $Venv ..."
    & py -3 -m venv $Venv
}

# --- Upgrade pip/setuptools/wheel ---------------------------------------
Write-Host "Upgrading pip/setuptools/wheel ..."
& $PyExe -m pip install --upgrade pip setuptools wheel

# --- Install requirements if exists -------------------------------------
if (Test-Path $Req) {
    Write-Host "Installing from requirements.txt ..."
    & $PyExe -m pip install -r $Req
} else {
    Write-Host "requirements.txt not found. Skipping dependency install."
}

# --- Log file ------------------------------------------------------------
$Ts = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $Logs "run_${Ts}.log"

# --- Run Python ----------------------------------------------------------
if (-not (Test-Path $PyScript)) {
    throw "Python script not found: $PyScript"
}

Write-Host "Running: $PyScript"
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $PyExe
$psi.WorkingDirectory = $Root
$forward = $args -join " "
$psi.Arguments = "`"$PyScript`" $forward"
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true
$psi.UseShellExecute = $false

$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
$null   = $proc.Start()
$stdOut = $proc.StandardOutput.ReadToEnd()
$stdErr = $proc.StandardError.ReadToEnd()
$proc.WaitForExit()

$stdOut | Tee-Object -FilePath $LogPath -Append | Out-Host
if ($stdErr) {
    "=== STDERR ===" | Tee-Object -FilePath $LogPath -Append | Out-Null
    $stdErr | Tee-Object -FilePath $LogPath -Append | Out-Host
}

# --- Compatibility mapping: search wf_* anywhere under data\analysis -----
$report = Join-Path $Root "report.md"
$anaDir = Join-Path $Root "data\analysis"

function _mapCopy($pattern, $destName) {
    $srcItem = $null
    if (Test-Path $anaDir) {
        $srcItem = Get-ChildItem -Path $anaDir -Filter $pattern -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 1
    }
    if ($srcItem) {
        $dst = Join-Path $Root $destName
        Copy-Item $srcItem.FullName $dst -Force
        "mapped: $($srcItem.FullName) -> $destName" | Tee-Object -FilePath $LogPath -Append | Out-Null
        return $true
    } else {
        "source not found (skip): $pattern" | Tee-Object -FilePath $LogPath -Append | Out-Null
        return $false
    }
}

# 必須3点を探して compare_* にコピー
$ok1 = _mapCopy "wf_summary_table.csv"   "compare_summary.csv"
$ok2 = _mapCopy "wf_summary_bar.png"     "compare_cum.png"
$ok3 = _mapCopy "wf_summary_scatter.png" "compare_monthly_mean.png"

# 最低限の report.md を生成（互換暫定）
$lines = @(
    "# Strategy Report", "",
    "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')", ""
)
if (Test-Path (Join-Path $Root "compare_cum.png")) {
    $lines += "## Cumulative Return"
    $lines += "![compare_cum](compare_cum.png)"
    $lines += ""
}
if (Test-Path (Join-Path $Root "compare_monthly_mean.png")) {
    $lines += "## Monthly Mean Return"
    $lines += "![compare_monthly_mean](compare_monthly_mean.png)"
    $lines += ""
}
if (Test-Path (Join-Path $Root "compare_summary.csv")) {
    $lines += "## Summary (CSV)"
    $lines += "- compare_summary.csv"
    $lines += ""
}
Set-Content -Path $report -Value $lines -Encoding UTF8
"wrote report.md (compat)" | Tee-Object -FilePath $LogPath -Append | Out-Null

# --- Post-run checks -----------------------------------------------------
$Expected = @(
    "compare_summary.csv",
    "compare_cum.png",
    "compare_monthly_mean.png",
    "report.md"
) | ForEach-Object { Join-Path $Root $_ }

$Expected = @(
    "compare_summary.csv",
    "compare_cum.png",
    "compare_monthly_mean.png",
    "report.md"
) | ForEach-Object { Join-Path $Root $_ }

$Missing = @($Expected | Where-Object { -not (Test-Path $_) })
if ($Missing.Count -gt 0) {
    "Missing outputs:`n - " + ($Missing -join "`n - ") | Tee-Object -FilePath $LogPath -Append | Out-Host
    Write-Host "Completed with missing artifacts. See log: $LogPath"
    exit 2
}

# --- Log rotation: keep 30 days -----------------------------------------
$Cutoff = (Get-Date).AddDays(-30)
Get-ChildItem $Logs -File -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt $Cutoff } |
    Remove-Item -Force -ErrorAction SilentlyContinue

Write-Host "Done. Log: $LogPath"
exit $proc.ExitCode
# ===== end =====
