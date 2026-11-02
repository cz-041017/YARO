<#
.SYNOPSIS
  Probe which Python package index is reachable from this machine.

.DESCRIPTION
  PyPI is blocked or TLS-inspected on many corporate networks; the symptom is
  "SSLError ... UNEXPECTED_EOF_WHILE_READING" or endless retries. This script
  tries a few well-known mirrors (plus any -IndexUrl you add) with a single
  tiny download and tells you which one works.

  It only downloads the "setuptools" wheel into a temp folder; it does not
  install anything and does not touch your environment.

  NOTE: intentionally ASCII-only (Windows PowerShell 5.1 reads .ps1 as ANSI).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\probe_index.ps1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\probe_index.ps1 -Extra @("https://my.mirror/simple")
#>
param(
    [string]$Python = "",
    [string[]]$Extra = @(),
    [int]$TimeoutSeconds = 10
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$probeDir = Join-Path $root "dist\_probe"
New-Item -ItemType Directory -Force -Path $probeDir | Out-Null

if (-not $Python) {
    $candidate = Join-Path $root ".venv-build\Scripts\python.exe"
    if (-not (Test-Path $candidate)) { $candidate = Join-Path $root ".venv\Scripts\python.exe" }
    if (Test-Path $candidate) { $Python = $candidate } else { $Python = "python" }
}

$indexes = @(
    "https://pypi.org/simple",
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple/",
    "https://mirrors.ustc.edu.cn/pypi/simple",
    "https://mirror.sjtu.edu.cn/pypi/web/simple"
) + $Extra

Write-Host "python: $Python"
Write-Host "proxy : HTTPS_PROXY=$($env:HTTPS_PROXY)  HTTP_PROXY=$($env:HTTP_PROXY)"
Write-Host ""

$ok = @()
foreach ($index in $indexes) {
    Write-Host -NoNewline ("probing {0} ... " -f $index)
    Get-ChildItem $probeDir -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    & $Python -m pip download --no-deps --disable-pip-version-check --retries 0 `
        --timeout $TimeoutSeconds --dest $probeDir --index-url $index setuptools *> $null
    $files = Get-ChildItem $probeDir -Filter *.whl -ErrorAction SilentlyContinue
    if ($LASTEXITCODE -eq 0 -and $files.Count -gt 0) {
        Write-Host "OK" -ForegroundColor Green
        $ok += $index
    } else {
        Write-Host "FAIL" -ForegroundColor DarkGray
    }
}

Write-Host ""
if ($ok.Count -eq 0) {
    Write-Host "No reachable index." -ForegroundColor Yellow
    Write-Host "Options:"
    Write-Host "  - use an HTTP proxy:  `$env:HTTPS_PROXY='http://host:port'; `$env:HTTP_PROXY='http://host:port'"
    Write-Host "  - copy wheels from another machine and use -FindLinks (see packaging\README.md)"
    Write-Host "  - build the exe on a machine that has working network access"
    exit 1
}

Write-Host "Working index(es):" -ForegroundColor Green
foreach ($index in $ok) { Write-Host "  $index" }
Write-Host ""
Write-Host "Next command:" -ForegroundColor Cyan
Write-Host ("  powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -IndexUrl {0}" -f $ok[0])
exit 0
