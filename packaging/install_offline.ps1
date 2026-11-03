<#
.SYNOPSIS
  Offline one-shot installer for colleagues: install YARO from a local wheelhouse,
  no network, and create a launcher + desktop shortcut.

.DESCRIPTION
  Use case: the target machine has Python 3.10+ but no usable network / proxy,
  so pip cannot reach PyPI.

  Prerequisite: on the build machine run packaging\build_wheelhouse.ps1 and copy
  dist\wheelhouse over together with the repo.

  What it does:
    1. check the Python version;
    2. create an isolated venv under -Dest (does not touch the existing environment);
    3. install yaro[full] fully offline from the wheelhouse;
    4. generate start-yaro.cmd and a desktop shortcut;
    5. initialize the workspace (demo data, CONTRACT.md, .env template).

  If the target machine has no Python at all: do not use this script, give them
  the packaged YARO.exe instead.

  NOTE: intentionally ASCII-only; Windows PowerShell 5.1 reads .ps1 as ANSI
  unless the file has a UTF-8 BOM.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\install_offline.ps1 -Wheelhouse D:\YARO\dist\wheelhouse
#>
param(
    [string]$Wheelhouse = (Join-Path (Split-Path -Parent $PSScriptRoot) "dist\wheelhouse"),
    [string]$Dest = (Join-Path $env:LOCALAPPDATA "YARO"),
    [string]$Python = "python",
    [switch]$NoShortcut
)

$ErrorActionPreference = "Stop"

Write-Host "[1/5] check Python" -ForegroundColor Cyan
try {
    $version = & $Python -c "import sys; print('%d.%d' % sys.version_info[:2])"
} catch {
    throw "no usable Python found. Use the packaged YARO.exe instead (no Python needed)."
}
if (-not $version) { throw "cannot determine the Python version. Use YARO.exe instead." }
$parts = $version.Split(".")
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 10)) {
    throw "Python 3.10+ is required, found $version. Use the packaged YARO.exe instead."
}
Write-Host "      Python $version"

if (-not (Test-Path $Wheelhouse)) {
    throw "wheel folder not found: $Wheelhouse. Copy dist\wheelhouse from the build machine first."
}
$wheelCount = (Get-ChildItem $Wheelhouse -Filter *.whl -ErrorAction SilentlyContinue).Count
if ($wheelCount -eq 0) { throw "no .whl files inside $Wheelhouse" }
Write-Host "      found $wheelCount wheel(s)"

Write-Host "[2/5] create isolated venv: $Dest\.venv" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$venv = Join-Path $Dest ".venv"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    & $Python -m venv $venv
}
$venvPy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "failed to create venv: $venvPy" }

Write-Host "[3/5] offline install: yaro[full]" -ForegroundColor Cyan
& $venvPy -m pip install --no-index --find-links $Wheelhouse --upgrade pip
& $venvPy -m pip install --no-index --find-links $Wheelhouse "yaro[full]"
if ($LASTEXITCODE -ne 0) {
    throw "offline install failed (exit $LASTEXITCODE). Usual cause: wheel Python version does not match this machine."
}

Write-Host "[4/5] create launcher" -ForegroundColor Cyan
$workspace = Join-Path $Dest "workspace"
$launcher = Join-Path $Dest "start-yaro.cmd"
@"
@echo off
setlocal
cd /d "%~dp0"
"$venvPy" -m yaro app --port 8765
if errorlevel 1 (
  echo.
  echo Window mode failed, falling back to browser mode...
  "$venvPy" -m yaro gui --port 8765
)
if errorlevel 1 pause
"@ | Set-Content -Path $launcher -Encoding ASCII
Write-Host "      -> $launcher"

if (-not $NoShortcut) {
    try {
        $desktop = [Environment]::GetFolderPath("Desktop")
        $shortcutPath = Join-Path $desktop "YARO.lnk"
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut($shortcutPath)
        $shortcut.TargetPath = $launcher
        $shortcut.WorkingDirectory = $Dest
        $shortcut.Description = "YARO visual development assistant"
        $shortcut.Save()
        Write-Host "      -> $shortcutPath"
    } catch {
        Write-Host "      desktop shortcut failed (not fatal): $_" -ForegroundColor DarkYellow
    }
}

Write-Host "[5/5] initialize workspace: $workspace" -ForegroundColor Cyan
& $venvPy -m yaro init --workspace $workspace
if ($LASTEXITCODE -ne 0) {
    Write-Host "      init returned non-zero (probably already initialized), safe to ignore" -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "Done. Double-click the YARO shortcut on the desktop, or run: $launcher" -ForegroundColor Green
Write-Host "Workspace: $workspace (sessions, demo data and artifacts live here)"
Write-Host "Try demo mode first: tick 'demo' in the composer, Ctrl+Enter (no tokens, no network)."
