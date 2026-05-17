param(
    [string]$PythonBin = "python"
)

# This is the recommended Windows setup route for the marker and student. It
# creates `.venv`, installs the pinned requirements, and checks imports so later
# README commands do not rely on whatever packages happen to be installed on the
# system Python.
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not (Test-Path "requirements.txt") -or -not (Test-Path "src") -or -not (Test-Path "app")) {
    throw "Run this script from the repository root."
}

& $PythonBin -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.11 or 3.12 is required for this pinned artefact environment."
}

Write-Host "Creating local virtual environment in .venv"
& $PythonBin -m venv .venv
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create .venv. Ensure Python venv support is installed."
}

$venvPython = Join-Path ".venv" "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = Join-Path ".venv" "bin/python"
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}

& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install requirements.txt."
}

$verifyImports = @"
required = ["flask", "joblib", "numpy", "openpyxl", "pandas", "pytest", "sklearn", "yaml"]
missing = []
for package in required:
    try:
        __import__(package)
    except Exception:
        missing.append(package)
if missing:
    raise SystemExit("Missing imports after setup: " + ", ".join(missing))
print("PASS: dependency imports verified")
"@
$verifyImports | & $venvPython -
if ($LASTEXITCODE -ne 0) {
    throw "FAIL: dependencies installed but required imports failed."
}

Write-Host ""
Write-Host "Optional local tooling for full artefact verification:"
Write-Host "- Python 3.11 or 3.12: https://www.python.org/downloads/"
Write-Host "- Docker Desktop: winget install Docker.DockerDesktop"
Write-Host "- Kind: winget install Kubernetes.kind"
Write-Host "- kubectl: winget install Kubernetes.kubectl"
Write-Host "- GitHub CLI: winget install GitHub.cli"
Write-Host ""
Write-Host "If PowerShell blocks scripts, run one of:"
Write-Host "  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/check_setup.ps1"
Write-Host ""
Write-Host "After installation, run:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/check_setup.ps1"
Write-Host ""
Write-Host "Activate the project environment before running README commands:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "Or call Python directly:"
Write-Host "  .\.venv\Scripts\python.exe -m pytest -q"
Write-Host ""
Write-Host "PASS: local PowerShell setup completed."
