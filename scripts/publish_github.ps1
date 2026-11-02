# Publish YARO to GitHub (ASCII only on purpose).
#
# Why ASCII: Windows PowerShell 5.1 reads .ps1 with the system ANSI code page when
# the file has no UTF-8 BOM, so non-ASCII text breaks parsing with confusing errors.
# Keep this file ASCII; put Chinese documentation in .md files.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\publish_github.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts\publish_github.ps1
#
# Prerequisites:
#   1. git installed:            git --version
#   2. SSH key registered:       ssh -T git@github.com   (should say "Hi <user>!")
#      If not: ssh-keygen -t ed25519 -C "you@example.com"  then paste
#      ~/.ssh/id_ed25519.pub into GitHub -> Settings -> SSH and GPG keys.
#   3. Empty repo created at https://github.com/cz-041017/YARO (no README needed).

param(
    [string]$Remote = "git@github.com:cz-041017/YARO.git",
    [string]$Branch = "main",
    [string]$Message = "Initial commit: YARO - visual development harness",
    [string]$AuthorName = "",
    [string]$AuthorEmail = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Host "repo: $root"

# ---------------------------------------------------------------- 1. git check
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {
    Write-Host "ERROR: git not found in PATH. Install Git for Windows first." -ForegroundColor Red
    Write-Host "       https://git-scm.com/download/win"
    exit 1
}
Write-Host ("git: " + (& git --version))

# ---------------------------------------------------------------- 2. init
if (-not (Test-Path (Join-Path $root ".git"))) {
    Write-Host "[1/6] git init -b $Branch"
    & git init -b $Branch | Out-Null
} else {
    Write-Host "[1/6] .git already exists, reusing it"
}

# ---------------------------------------------------------------- 3. stage
Write-Host "[2/6] git add -A"
& git add -A
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: git add failed" -ForegroundColor Red; exit 1 }

$staged = @(& git diff --cached --name-only)
$count = $staged.Count
Write-Host ("      staged files: " + $count)
if ($count -eq 0) {
    Write-Host "Nothing to commit." -ForegroundColor Yellow
    if ($DryRun) { exit 0 }
} else {
    $staged | Sort-Object | Select-Object -First 30 | ForEach-Object { Write-Host ("      " + $_) }
    if ($count -gt 30) { Write-Host ("      ... and " + ($count - 30) + " more") }
}

# ------------------------------------------------------------ 4. safety audit
Write-Host "[3/6] safety audit (secrets / runtime data / build output)"
$forbidden = $staged | Where-Object {
    $_ -match '(^|/)\.env$' -or
    $_ -match 'secrets\.json$' -or
    $_ -match '(^|/)\.yaro/' -or
    $_ -match '(^|/)\.venv' -or
    $_ -match '(^|/)dist/' -or
    $_ -match '(^|/)build/' -or
    $_ -match 'pytest-of-' -or
    $_ -match '(^|/)demo/' -or
    $_ -match '(^|/)artifacts/' -or
    $_ -match '(^|/)sessions/' -or
    $_ -match '(^|/)runs/'
}
if ($forbidden) {
    Write-Host "ERROR: refusing to publish, these files must stay local:" -ForegroundColor Red
    $forbidden | ForEach-Object { Write-Host ("      " + $_) -ForegroundColor Red }
    Write-Host "Fix .gitignore, then run: git rm -r --cached . ; git add -A"
    exit 2
}
Write-Host "      clean: no secrets / runtime data / build output staged"

$big = @()
foreach ($file in $staged) {
    if (Test-Path -LiteralPath $file) {
        $size = (Get-Item -LiteralPath $file).Length
        if ($size -gt 5MB) { $big += ("{0} ({1:N1} MB)" -f $file, ($size / 1MB)) }
    }
}
if ($big.Count -gt 0) {
    Write-Host "WARNING: files larger than 5 MB (GitHub rejects >100 MB):" -ForegroundColor Yellow
    $big | ForEach-Object { Write-Host ("      " + $_) -ForegroundColor Yellow }
}

if ($DryRun) {
    Write-Host "[DryRun] stopping before commit. Nothing was committed or pushed." -ForegroundColor Yellow
    exit 0
}

# ---------------------------------------------------------------- 5. identity
if ($AuthorName)  { & git config user.name  $AuthorName }
if ($AuthorEmail) { & git config user.email $AuthorEmail }
$currentName  = (& git config user.name) 2>$null
$currentEmail = (& git config user.email) 2>$null
if (-not $currentName -or -not $currentEmail) {
    Write-Host "ERROR: git identity not configured." -ForegroundColor Red
    Write-Host "       git config --global user.name  `"Your Name`""
    Write-Host "       git config --global user.email `"you@example.com`""
    Write-Host "       (or rerun this script with -AuthorName / -AuthorEmail)"
    exit 3
}

# ---------------------------------------------------------------- 6. commit
if ($count -gt 0) {
    Write-Host "[4/6] git commit"
    & git commit -m $Message | Out-Host
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: git commit failed" -ForegroundColor Red; exit 1 }
} else {
    Write-Host "[4/6] no changes to commit"
}

# ---------------------------------------------------------------- 7. remote
$existing = (& git remote get-url origin) 2>$null
if ($LASTEXITCODE -eq 0 -and $existing) {
    if ($existing -ne $Remote) {
        Write-Host ("[5/6] update origin: " + $existing + " -> " + $Remote)
        & git remote set-url origin $Remote
    } else {
        Write-Host "[5/6] origin already points to $Remote"
    }
} else {
    Write-Host "[5/6] git remote add origin $Remote"
    & git remote add origin $Remote
}

# ---------------------------------------------------------------- 8. push
Write-Host "[6/6] git push -u origin $Branch"
& git push -u origin $Branch
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Push failed. Most common causes:" -ForegroundColor Red
    Write-Host "  1. SSH key not registered:   ssh -T git@github.com"
    Write-Host "     create key:               ssh-keygen -t ed25519 -C `"you@example.com`""
    Write-Host "     then add ~/.ssh/id_ed25519.pub to GitHub (Settings -> SSH and GPG keys)"
    Write-Host "  2. Remote repo not created:  https://github.com/new  (name: YARO, empty)"
    Write-Host "  3. Wrong account:            ssh -T git@github.com prints the user name"
    Write-Host "  4. Non-empty remote:         git pull --rebase origin $Branch  then push again"
    Write-Host "  HTTPS alternative:           git remote set-url origin https://github.com/cz-041017/YARO.git"
    exit 4
}

Write-Host ""
Write-Host "Done. Open: https://github.com/cz-041017/YARO" -ForegroundColor Green
& git log --oneline -1
