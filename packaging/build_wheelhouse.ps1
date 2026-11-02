<#
.SYNOPSIS
  Download YARO and all of its dependencies as local wheels (on a machine with network).

.DESCRIPTION
  Use case: a colleague has Python but no network / blocked proxy, so pip cannot
  reach PyPI. Build the wheelhouse here, copy it over, then install offline.

  Output: dist\wheelhouse\*.whl

  On the target machine:
      python -m pip install --no-index --find-links dist\wheelhouse "yaro[full]"
      python -m yaro init
      python -m yaro app

  The wheelhouse is tied to the target machine's Python version / architecture.
  Default is Python 3.11 + win_amd64. If the colleague runs 3.12, rerun with
  -PythonVersion 312.

  NOTE: intentionally ASCII-only; Windows PowerShell 5.1 reads .ps1 as ANSI
  unless the file has a UTF-8 BOM.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\build_wheelhouse.ps1
#>
param(
    [string]$Python = "python",
    [string]$PythonVersion = "311",
    [string]$Platform = "win_amd64"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$wheelhouse = Join-Path $root "dist\wheelhouse"
New-Item -ItemType Directory -Force -Path $wheelhouse | Out-Null

Write-Host "[1/3] build yaro's own wheel" -ForegroundColor Cyan
& $Python -m pip wheel . --no-deps -w $wheelhouse
if ($LASTEXITCODE -ne 0) { throw "failed to build the yaro wheel" }

Write-Host "[2/3] download dependency wheels (target: cp$PythonVersion / $Platform)" -ForegroundColor Cyan
$packages = @(
    "pywebview>=5.0",
    "openai>=1.40",
    "numpy>=1.26",
    "opencv-python-headless>=4.9"
)
& $Python -m pip download `
    --only-binary=:all: `
    --dest $wheelhouse `
    --platform $Platform `
    --python-version $PythonVersion `
    --implementation cp `
    @packages
if ($LASTEXITCODE -ne 0) { throw "failed to download dependencies (check network / proxy)" }

Write-Host "[3/3] summary" -ForegroundColor Cyan
$wheels = Get-ChildItem $wheelhouse -Filter *.whl
Write-Host ("      {0} wheel(s) -> {1}" -f $wheels.Count, $wheelhouse)
Write-Host ""
Write-Host "Copy the whole dist\wheelhouse folder together with the repo (USB / intranet share)." -ForegroundColor Green
