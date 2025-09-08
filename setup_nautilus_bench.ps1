# Requires: Windows PowerShell, Python 3.11, internet access
# Purpose: Create a working folder `nautilus_bench`, set up venv, 
#          install minimal deps, clone nautilus_trader, and pip install -e .
# Usage: Right-click -> Run with PowerShell, or:
#   PowerShell -ExecutionPolicy Bypass -File .\setup_nautilus_bench.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Step($msg) { Write-Host "[>] $msg" -ForegroundColor Cyan }
function Write-Info($msg) { Write-Host "[i] $msg" -ForegroundColor Gray }
function Write-Ok($msg)   { Write-Host "[✓] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[!] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[x] $msg" -ForegroundColor Red }

function Test-Command($name) {
  $old = $ErrorActionPreference; $ErrorActionPreference = 'SilentlyContinue'
  try { Get-Command $name | Out-Null; return $true } catch { return $false } finally { $ErrorActionPreference = $old }
}

# 1) Create working folder
$Base = Join-Path $HOME 'nautilus_bench'
Write-Step "Create folder: $Base"
New-Item -ItemType Directory -Force -Path $Base | Out-Null
Write-Ok "Folder ready"

# 2) Create and activate venv (Python 3.11)
Write-Step "Ensure Python 3.11 and create venv"
$pyInvocation = 'py -3.11'
$pythonOk = $false
try {
  $ver = & py -3.11 -V 2>$null
  if ($LASTEXITCODE -eq 0 -and $ver -match 'Python 3\.11') { $pythonOk = $true }
} catch {}

if (-not $pythonOk) {
  if (Test-Command 'python') {
    $ver = & python -V
    if ($ver -match 'Python 3\.11') { $pyInvocation = 'python'; $pythonOk = $true }
  }
}

if (-not $pythonOk) {
  Write-Err 'Python 3.11 not found. Install Python 3.11 then re-run.'
  Write-Info 'Install via Microsoft Store (Python 3.11) or winget: winget install -e --id Python.Python.3.11'
  exit 1
}

$VenvPath = Join-Path $Base '.venv'
if (-not (Test-Path $VenvPath)) {
  & $pyInvocation -m venv $VenvPath
  Write-Ok "Venv created at $VenvPath"
} else {
  Write-Info "Venv already exists: $VenvPath"
}

Write-Step 'Activate venv'
& (Join-Path $VenvPath 'Scripts/Activate.ps1')
Write-Ok "Venv activated: $env:VIRTUAL_ENV"

# 3) Upgrade pip/setuptools/wheel and install minimal dependencies
Write-Step 'Upgrade pip and install packages (pandas, pyarrow, psutil)'
python -m pip install --upgrade pip setuptools wheel
pip install pandas pyarrow psutil
Write-Ok 'Base Python packages installed'

# 4) Ensure Git is available
Write-Step 'Check Git availability'
if (-not (Test-Command 'git')) {
  Write-Err 'Git is not installed. Install Git for Windows and re-run.'
  Write-Info 'Install via winget: winget install -e --id Git.Git'
  exit 1
} else {
  git --version
  Write-Ok 'Git is available'
}

# 5) Clone nautilus_trader and install editable
Set-Location $Base
if (-not (Test-Path (Join-Path $Base 'nautilus_trader'))) {
  Write-Step 'Clone nautilus_trader from GitHub'
  git clone https://github.com/nautechsystems/nautilus_trader.git
  Write-Ok 'Repository cloned'
} else {
  Write-Info 'Repository already exists; pulling latest changes'
  Push-Location (Join-Path $Base 'nautilus_trader')
  git pull --ff-only
  Pop-Location
}

Set-Location (Join-Path $Base 'nautilus_trader')

Write-Step 'pip install -e . (editable install)'
try {
  pip install -e .
  Write-Ok 'nautilus_trader installed (editable)'
}
catch {
  Write-Err 'Editable install failed.'
  Write-Warn 'Common causes: Rust toolchain not installed, or missing Build Tools.'
  Write-Info 'Fix steps:'
  Write-Info '  1) Install Rust toolchain (stable): winget install -e --id Rustlang.Rustup'
  Write-Info '     Then in a new terminal: rustup default stable; rustup update'
  Write-Info '  2) Install Visual Studio 2022 Build Tools (C++):'
  Write-Info '     winget install -e --id Microsoft.VisualStudio.2022.BuildTools'
  Write-Info '     Select "Desktop development with C++" workload (incl. MSVC, CMake, Windows SDK)'
  Write-Info '  3) Re-open terminal, re-activate venv, then rerun: pip install -e .'
  Write-Info '     Commands to retry:'
  Write-Info "       `& $pyInvocation -m venv $VenvPath; & $VenvPath\Scripts\Activate.ps1`"
  Write-Info '       cd $HOME\nautilus_bench\nautilus_trader'
  Write-Info '       pip install -e .'
  throw
}

Write-Ok 'All steps completed successfully.'

