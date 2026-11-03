# Publish YARO to GitHub (ASCII only on purpose).
#
# Why ASCII: Windows PowerShell 5.1 reads .ps1 with the system ANSI code page when
# the file has no UTF-8 BOM, so non-ASCII text breaks parsing with confusing errors.
# Keep this file ASCII; put Chinese documentation in .md files.
#
# Why $ErrorActionPreference = "Continue": native commands (git) write progress and
# hints to stderr. With "Stop", any stderr line from git becomes a terminating
# NativeCommandError (that is what broke the first version at `git remote get-url`).
# So: never let stderr kill us; check $LASTEXITCODE explicitly after every git call.
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

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Write-Host "repo: $root"

# Run a git command; return its captured output lines (stdout+stderr merged) and
# expose the exit code via the script-scope variable $GitExit.
function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArgs)
    $output = & git @GitArgs 2>&1
    $script:GitExit = $LASTEXITCODE
    return @($output)
}

function Write-GitOutput {
    param([string[]]$Lines)
    foreach ($line in $Lines) { Write-Host ("      " + $line) }
}

# ---------------------------------------------------------------- 1. git check
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {
    Write-Host "ERROR: git not found in PATH. Install Git for Windows first." -ForegroundColor Red
    Write-Host "       https://git-scm.com/download/win"
    exit 1
}
$version = Invoke-Git --version
Write-Host ("git: " + ($version -join " "))

# ---------------------------------------------------------------- 2. init
if (-not (Test-Path (Join-Path $root ".git"))) {
    Write-Host "[1/6] git init -b $Branch"
    $out = Invoke-Git init -b $Branch
    if ($script:GitExit -ne 0) { Write-GitOutput $out; Write-Host "ERROR: git init failed" -ForegroundColor Red; exit 1 }
} else {
    Write-Host "[1/6] .git already exists, reusing it"
}

# ---------------------------------------------------------------- 3. stage
Write-Host "[2/6] git add -A"
$out = Invoke-Git add -A
if ($script:GitExit -ne 0) { Write-GitOutput $out; Write-Host "ERROR: git add failed" -ForegroundColor Red; exit 1 }

$out = Invoke-Git diff --cached --name-only
if ($script:GitExit -ne 0) { Write-GitOutput $out; Write-Host "ERROR: cannot list staged files" -ForegroundColor Red; exit 1 }
$staged = @($out | Where-Object { $_ -ne "" -and $_ -notmatch '^warning:' })
$count = $staged.Count
Write-Host ("      staged files: " + $count)
if ($count -gt 0) {
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
if ($AuthorName)  { Invoke-Git config user.name  $AuthorName  | Out-Null }
if ($AuthorEmail) { Invoke-Git config user.email $AuthorEmail | Out-Null }
$nameOut = Invoke-Git config user.name
$currentName = ($nameOut -join "").Trim()
$mailOut = Invoke-Git config user.email
$currentEmail = ($mailOut -join "").Trim()
if (-not $currentName -or -not $currentEmail) {
    Write-Host "ERROR: git identity not configured." -ForegroundColor Red
    Write-Host "       git config --global user.name  `"Your Name`""
    Write-Host "       git config --global user.email `"you@example.com`""
    Write-Host "       (or rerun this script with -AuthorName / -AuthorEmail)"
    exit 3
}
Write-Host ("      author: " + $currentName + " <" + $currentEmail + ">")

# ---------------------------------------------------------------- 6. commit
if ($count -gt 0) {
    Write-Host "[4/6] git commit"
    $out = Invoke-Git commit -m $Message
    Write-GitOutput $out
    if ($script:GitExit -ne 0) { Write-Host "ERROR: git commit failed" -ForegroundColor Red; exit 1 }
} else {
    $headOut = Invoke-Git rev-parse --verify HEAD
    if ($script:GitExit -eq 0) {
        Write-Host "[4/6] working tree clean, nothing to commit"
    } else {
        Write-Host "[4/6] nothing staged and no commit yet: check .gitignore" -ForegroundColor Yellow
        exit 1
    }
}

# ---------------------------------------------------------------- 7. remote
Write-Host "[4.5/6] remote origin"
$listOut = Invoke-Git remote
$remotes = @($listOut | Where-Object { $_ -ne "" })
if ($remotes -contains "origin") {
    $urlOut = Invoke-Git remote get-url origin
    $existing = ("$urlOut").Trim()
    if ($script:GitExit -eq 0 -and $existing -ne $Remote) {
        Write-Host ("      update origin: " + $existing + " -> " + $Remote)
        Invoke-Git remote set-url origin $Remote | Out-Null
    } else {
        Write-Host ("      origin already points to " + $Remote)
    }
} else {
    Write-Host ("      git remote add origin " + $Remote)
    $out = Invoke-Git remote add origin $Remote
    if ($script:GitExit -ne 0) { Write-GitOutput $out; Write-Host "ERROR: git remote add failed" -ForegroundColor Red; exit 1 }
}

# ---------------------------------------------------------------- 8. push
Write-Host "[5/6] git push -u origin $Branch"
$out = Invoke-Git push -u origin $Branch
Write-GitOutput $out
if ($script:GitExit -ne 0) {
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
Write-Host "[6/6] Done. Open: https://github.com/cz-041017/YARO" -ForegroundColor Green
$logOut = Invoke-Git log --oneline -1
Write-GitOutput $logOut
exit 0
