<#
.SYNOPSIS
  Build a distributable YARO package on a Windows x64 build machine (needs a package index).

.DESCRIPTION
  Outputs:
    dist\YARO\               app folder (Python runtime + deps + web assets + CONTRACT.md)
    dist\YARO\YARO.exe       double-click to start (console + app window)
    dist\YARO-win64.zip      send this zip to colleagues

  A colleague only needs to unzip and double-click YARO.exe.
  No Python, no pip, no network, no browser settings on the target machine.

  Build machine requirements: Windows x64, Python 3.10+, and a reachable package index.

  If PyPI is blocked (SSL / proxy / TLS inspection), pass a mirror explicitly:
      powershell -File packaging\build_windows.ps1 -IndexUrl https://pypi.tuna.tsinghua.edu.cn/simple
  or point at a local wheelhouse:
      powershell -File packaging\build_windows.ps1 -FindLinks D:\wheels
  Use packaging\probe_index.ps1 to find out which index actually answers.

  NOTE: this file is intentionally ASCII-only. Windows PowerShell 5.1 reads .ps1
  files as ANSI unless they have a UTF-8 BOM, so non-ASCII text here would break
  the parser on Chinese/Japanese Windows installations.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1

.EXAMPLE
  # corporate network needs a mirror
  powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -IndexUrl https://pypi.tuna.tsinghua.edu.cn/simple

.EXAMPLE
  # dependencies already installed, just repackage
  powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -SkipDeps
#>
param(
    [string]$Python = "python",
    [switch]$SkipDeps,
    [switch]$NoZip,
    # Editable/local installs build in an isolated env, which tries to download
    # setuptools. On a locked-down network that download is often what fails;
    # -NoIsolation reuses the toolchain already inside the build venv.
    [switch]$NoIsolation,
    [string]$IndexUrl = "",
    [string[]]$TrustedHost = @(),
    [string]$FindLinks = "",
    # Reuse an existing venv (e.g. the repo's .venv) instead of creating .venv-build.
    # Useful when pip cannot install into a fresh venv on this machine:
    #   powershell -File packaging\build_windows.ps1 -BuildVenv .venv -SkipDeps
    [string]$BuildVenv = ".venv-build",
    # Bundle OpenCV/numpy too (open the exe's calibration tools). Bigger zip.
    [switch]$WithCv
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Package-index settings are applied as environment variables on purpose: pip's
# build-isolation subprocess must inherit them too, and flags would not reach it.
if ($IndexUrl) { $env:PIP_INDEX_URL = $IndexUrl }
if ($TrustedHost.Count -gt 0) { $env:PIP_TRUSTED_HOST = ($TrustedHost -join " ") }
if ($FindLinks) { $env:PIP_FIND_LINKS = $FindLinks }

function Show-NetworkHelp {
    Write-Host ""
    Write-Host "Package install failed. If PyPI is blocked here, try one of:" -ForegroundColor Yellow
    Write-Host "  1) find a reachable mirror:      powershell -File packaging\probe_index.ps1"
    Write-Host "  2) build with a mirror:          powershell -File packaging\build_windows.ps1 -IndexUrl <mirror>"
    Write-Host "  3) behind an HTTP proxy:         `$env:HTTPS_PROXY='http://host:port'; `$env:HTTP_PROXY='http://host:port'"
    Write-Host "  4) no network at all:            build packaging\build_wheelhouse.ps1 on a networked machine,"
    Write-Host "                                   copy dist\wheelhouse here, then use -FindLinks dist\wheelhouse"
    Write-Host "  5) or just build the package on a machine that does have network access."
    Write-Host ""
}

function Install-Deps {
    param([string]$VenvPy, [bool]$WithCv)
    # Install runtime packages BY NAME, never as "-e .": PyInstaller reads yaro
    # straight from the source tree (see pathex in yaro.spec), so an editable
    # install is not needed. That also avoids the build-isolation step, which is
    # what silently stalls pip 26 on some locked-down machines.
    $targets = @("--upgrade", "pywebview>=5.0", "openai>=1.40", "pyinstaller",
        # LangGraph runtime (v0.7+ core). langchain-openai is the DeepSeek base;
        # langchain-mcp-adapters pulls the `mcp` SDK for community MCP tools.
        "langgraph>=0.2", "langchain-core>=0.3", "langchain-openai>=0.2", "langchain-mcp-adapters>=0.1")
    if ($WithCv) { $targets += @("numpy>=1.26", "opencv-python-headless>=4.9") }
    $shown = "pip install " + ($targets -join " ")
    Write-Host "      > $shown" -ForegroundColor DarkGray
    # --no-cache-dir: a half-killed earlier pip can leave a cache lock behind and
    # the next run then waits forever without printing anything.
    # --progress-bar off: one line per download instead of carriage-return redraws.
    & $VenvPy -m pip install --no-cache-dir --disable-pip-version-check --progress-bar off @targets
    return $LASTEXITCODE
}

$venvPy = Join-Path $root (Join-Path $BuildVenv "Scripts\python.exe")

Write-Host "[1/6] prepare build venv: $BuildVenv" -ForegroundColor Cyan
if (-not (Test-Path $venvPy)) {
    & $Python -m venv $BuildVenv
}
if (-not (Test-Path $venvPy)) {
    throw "failed to create venv: $venvPy (is '$Python' a working Python 3.10+?)"
}

Write-Host "[2/6] install deps: pywebview + openai + pyinstaller" -ForegroundColor Cyan
if ($IndexUrl) { Write-Host "      index: $IndexUrl" -ForegroundColor DarkGray }
if ($FindLinks) { Write-Host "      find-links: $FindLinks" -ForegroundColor DarkGray }
if ($SkipDeps) {
    Write-Host "      skipped (-SkipDeps)"
} else {
    Write-Host "      this downloads pywebview / openai / pyinstaller (a few tens of MB)" -ForegroundColor DarkGray
    Write-Host "      it can take a few minutes; each finished download prints one line below" -ForegroundColor DarkGray
    Write-Host "      > pip install --progress-bar off --upgrade pip" -ForegroundColor DarkGray
    & $venvPy -m pip install --progress-bar off --disable-pip-version-check --upgrade pip
    $code = Install-Deps -VenvPy $venvPy -WithCv:$WithCv
    if ($code -ne 0) {
        Show-NetworkHelp
        throw "dependency install failed (exit $code)"
    }
}

# PyInstaller bundles yaro from the source tree (pathex in yaro.spec), so make
# sure this process can import the repo even without an editable install.
$env:PYTHONPATH = $root

Write-Host "[3/6] PyInstaller build" -ForegroundColor Cyan
& $venvPy -m PyInstaller --noconfirm --clean packaging\yaro.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit $LASTEXITCODE)" }

$exe = Join-Path $root "dist\YARO\YARO.exe"
if (-not (Test-Path $exe)) { throw "build output missing: $exe" }

Write-Host "[4/6] verify web assets and contract are inside the bundle" -ForegroundColor Cyan
$candidates = @(
    "dist\YARO\_internal\yaro\web\index.html",
    "dist\YARO\yaro\web\index.html"
)
$webOk = $false
foreach ($c in $candidates) { if (Test-Path $c) { $webOk = $true } }
if (-not $webOk) { throw "web assets missing from bundle (blank window). Check 'datas' in packaging\yaro.spec" }

$contractCandidates = @(
    "dist\YARO\_internal\yaro\data\CONTRACT.md",
    "dist\YARO\yaro\data\CONTRACT.md"
)
$contractOk = $false
foreach ($c in $contractCandidates) { if (Test-Path $c) { $contractOk = $true } }
if (-not $contractOk) { throw "CONTRACT.md missing from bundle. Check 'datas' in packaging\yaro.spec" }

Write-Host "[5/6] smoke test: YARO.exe doctor" -ForegroundColor Cyan
& $exe doctor
Write-Host "      JSON output above means the bundle is alive." -ForegroundColor DarkGray

Write-Host "[6/6] zip" -ForegroundColor Cyan
if ($NoZip) {
    Write-Host "      skipped (-NoZip)"
} else {
    $zip = Join-Path $root "dist\YARO-win64.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path (Join-Path $root "dist\YARO") -DestinationPath $zip
    Write-Host "      -> $zip"
}

Write-Host ""
Write-Host "Done. Send dist\YARO-win64.zip to colleagues; they unzip and double-click YARO.exe." -ForegroundColor Green
