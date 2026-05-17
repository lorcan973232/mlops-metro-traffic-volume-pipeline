param(
    [string]$PythonBin = "",
    [switch]$PythonOnly,
    [switch]$RequireGh
)

# This check is intentionally diagnostic rather than clever. It tells the student
# or marker whether Python, the virtual environment, Docker, Kind, kubectl, Git,
# and optionally GitHub CLI are ready before a live demo or local verification run.
$ErrorActionPreference = "Continue"
Set-StrictMode -Version Latest

$wingetLinks = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links"
if ((Test-Path $wingetLinks) -and ($env:PATH -notlike "*$wingetLinks*")) {
    $env:PATH = "$env:PATH;$wingetLinks"
}
$dockerBin = "C:\Program Files\Docker\Docker\resources\bin"
if ((Test-Path $dockerBin) -and ($env:PATH -notlike "*$dockerBin*")) {
    $env:PATH = "$env:PATH;$dockerBin"
}

$script:Blocked = $false
$script:Failed = $false

function Write-Check {
    param(
        [string]$Status,
        [string]$Name,
        [string]$Detail
    )
    Write-Host "${Status}: ${Name} - ${Detail}"
}

function Pass-Check {
    param([string]$Name, [string]$Detail)
    Write-Check "PASS" $Name $Detail
}

function Block-Check {
    param([string]$Name, [string]$Detail)
    $script:Blocked = $true
    Write-Check "BLOCKED_BY_LOCAL_SETUP" $Name $Detail
}

function Fail-Check {
    param([string]$Name, [string]$Detail)
    $script:Failed = $true
    Write-Check "FAIL" $Name $Detail
}

function Require-Command {
    # Missing local tools should be reported as setup blockers, not confused with
    # project failures. That distinction matters when marking reproducibility.
    param([string]$Name, [string]$Guidance)
    if (Test-Path $Name) {
        Pass-Check $Name (Resolve-Path $Name).Path
        return $true
    }
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        Block-Check $Name $Guidance
        return $false
    }
    Pass-Check $Name $command.Source
    return $true
}

if ((Test-Path "README.md") -and (Test-Path "requirements.txt") -and (Test-Path "src") -and (Test-Path "app")) {
    Pass-Check "repository root" (Get-Location).Path
} else {
    Fail-Check "repository root" "Run this script from the repository root."
}

Pass-Check "windows shell guidance" "PowerShell scripts are the recommended Windows path. Use Git Bash explicitly for .sh scripts: C:\Program Files\Git\bin\bash.exe. Do not use WSL bash unless WSL is configured."

$pythonCommand = $PythonBin
if (-not $pythonCommand) {
    if ($env:PYTHON_BIN) {
        $pythonCommand = $env:PYTHON_BIN
    } elseif (Test-Path ".venv\Scripts\python.exe") {
        $pythonCommand = ".venv\Scripts\python.exe"
    } elseif (Test-Path ".venv/bin/python") {
        $pythonCommand = ".venv/bin/python"
    } else {
        $pythonCommand = "python"
    }
}

$pythonAvailable = Require-Command $pythonCommand "Install Python 3.11 or 3.12: winget install Python.Python.3.12"
Require-Command "git" "Install Git: winget install Git.Git" | Out-Null
if ($RequireGh) {
    Require-Command "gh" "Install GitHub CLI: winget install GitHub.cli; then run: gh auth login" | Out-Null
}

if ($pythonAvailable) {
    $versionScript = "import sys, venv; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)"
    & $pythonCommand -c $versionScript | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $version = (& $pythonCommand --version) -join " "
        Pass-Check "python version and venv" $version
    } else {
        Fail-Check "python version and venv" "Use Python 3.11 or 3.12 with venv support."
    }

    & $pythonCommand -m pip --version | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $pipVersion = (& $pythonCommand -m pip --version) -join " "
        Pass-Check "pip" $pipVersion
    } else {
        Block-Check "pip" "Install pip with Python or run: $pythonCommand -m ensurepip --upgrade"
    }

    if (Test-Path ".venv") {
        Pass-Check "virtual environment" ".venv exists; selected interpreter is $pythonCommand"
    } else {
        Block-Check "virtual environment" "Run: powershell -ExecutionPolicy Bypass -File scripts/setup_local.ps1"
    }

    $dependencyScript = @"
required = ["flask", "joblib", "numpy", "openpyxl", "pandas", "pytest", "sklearn", "yaml"]
missing = []
for package in required:
    try:
        __import__(package)
    except Exception:
        missing.append(package)
if missing:
    raise SystemExit(", ".join(missing))
"@
    $dependencyScript | & $pythonCommand -
    if ($LASTEXITCODE -eq 0) {
        Pass-Check "python dependencies" "required packages import successfully"
    } else {
        Block-Check "python dependencies" "Run: $pythonCommand -m pip install -r requirements.txt"
    }
}

if ($PythonOnly) {
    Pass-Check "deployment tooling" "Docker, Kind, and kubectl checks skipped for non-deployment CI."
} else {
    $dockerAvailable = Require-Command "docker" "Install Docker Desktop: winget install Docker.DockerDesktop; then start Docker Desktop"
    Require-Command "kind" "Install Kind: winget install Kubernetes.kind" | Out-Null
    Require-Command "kubectl" "Install kubectl: winget install Kubernetes.kubectl" | Out-Null

    if ($dockerAvailable) {
        & docker info | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Pass-Check "docker daemon" "Docker daemon is running"
        } else {
            Block-Check "docker daemon" "Start Docker Desktop, then rerun scripts/check_setup.ps1"
        }
    }
}

if ($script:Failed) {
    Write-Host "FAIL: setup check failed. Fix failed checks above."
    exit 1
}
if ($script:Blocked) {
    Write-Host "BLOCKED_BY_LOCAL_SETUP: install or start the blocked dependencies above."
    exit 1
}

Write-Host "PASS: local setup check passed."
