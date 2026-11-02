# YARO one-shot installer (Windows / PowerShell 5.1+ or pwsh)
#
#   .\scripts\install.ps1              # install into .venv: cloud API + vision deps + desktop window
#   .\scripts\install.ps1 -Editable    # dev mode (edits take effect immediately)
#   .\scripts\install.ps1 -Extras cv   # vision deps only, no cloud API / no desktop window
#
# Principle: never install into the system Python; everything goes into .venv inside the repo.
# The desktop window (pywebview) matters for "works the same on everyone's machine": it renders
# through the system WebView, so browser extensions / forced dark mode cannot rewrite it.
# If it cannot be installed the script only warns (everything else still works via `yaro gui`).
#
# NOTE: intentionally ASCII-only. Windows PowerShell 5.1 reads .ps1 files as ANSI unless they
# have a UTF-8 BOM; non-ASCII text here would break the parser on Chinese/Japanese Windows.

param(
    [switch]$Editable,
    [string]$Extras = "full",
    [switch]$SkipInit,
    [switch]$NoDesktop
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv"
$python = Join-Path $venv "Scripts\python.exe"

Push-Location $root
try {
    if (-not (Test-Path $python)) {
        Write-Host "[1/5] create venv: .venv"
        python -m venv $venv
    } else {
        Write-Host "[1/5] reuse existing .venv"
    }

    Write-Host "[2/5] upgrade pip"
    & $python -m pip install --upgrade pip --quiet

    Write-Host "[3/5] install yaro[$Extras]"
    if ($Extras -ne "") {
        if ($Editable) { & $python -m pip install -e ".[$Extras]" } else { & $python -m pip install ".[$Extras]" }
    } else {
        if ($Editable) { & $python -m pip install -e . } else { & $python -m pip install . }
    }

    # Probe the desktop backend separately: a display-layer failure must not fail the install.
    $desktop = $false
    if (-not $NoDesktop) {
        Write-Host "[4/5] check desktop backend (yaro.desktop / pywebview)"
        & $python -c "from yaro.desktop import available; raise SystemExit(0 if available() else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $desktop = $true
            Write-Host "      available: start with: yaro app (recommended)"
        } else {
            Write-Host "      not available: will fall back to browser mode (yaro gui)" -ForegroundColor Yellow
            Write-Host "      to enable it manually: python -m pip install pywebview" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[4/5] desktop window skipped (-NoDesktop)"
    }

    if (-not $SkipInit) {
        Write-Host "[5/5] initialize workspace (idempotent)"
        & $python -m yaro init
    }

    Write-Host ""
    Write-Host "Done. Activate the environment before running commands:" -ForegroundColor Green
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  yaro doctor"
    if ($desktop) { Write-Host "  yaro app      # desktop window (recommended)" } else { Write-Host "  yaro gui      # browser mode" }
} finally {
    Pop-Location
}
