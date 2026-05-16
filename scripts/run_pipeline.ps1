param(
    [string]$PythonBin = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $PythonBin) {
    if ($env:PYTHON_BIN) {
        $PythonBin = $env:PYTHON_BIN
    } elseif (Test-Path ".venv\Scripts\python.exe") {
        $PythonBin = ".venv\Scripts\python.exe"
    } elseif (Test-Path ".venv/bin/python") {
        $PythonBin = ".venv/bin/python"
    } else {
        $PythonBin = "python"
    }
}

function Invoke-Stage {
    param(
        [string]$Name,
        [string[]]$Arguments
    )

    Write-Host "RUN: $Name"
    & $PythonBin @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stage failed: $Name"
    }
    Write-Host "PASS: $Name"
}

Invoke-Stage "compile Python modules" @("-m", "compileall", "app", "src", "tests", "scripts")
Invoke-Stage "data acquisition" @("-m", "src.data")
Invoke-Stage "data preprocessing" @("-m", "src.preprocess")
Invoke-Stage "model selection" @("-m", "src.model_selection")
Invoke-Stage "model training" @("-m", "src.train")
Invoke-Stage "model evaluation and quality gate" @("-m", "src.evaluate")
Invoke-Stage "model registry metadata" @("-m", "src.model_registry")
Invoke-Stage "prediction smoke path" @("-m", "src.predict")
Invoke-Stage "SHAP explainability" @("scripts/explain_model.py")
Invoke-Stage "fairness audit" @("scripts/fairness_audit.py")
Invoke-Stage "cost-benefit analysis" @("scripts/cost_benefit_analysis.py")
Invoke-Stage "offline monitoring" @("scripts/monitor.py")
Invoke-Stage "drift/data-quality check" @("scripts/check_drift.py")
Invoke-Stage "pytest suite" @("-m", "pytest", "-q")
Invoke-Stage "ruff lint" @("-m", "ruff", "check", "src", "tests", "scripts")
Invoke-Stage "Flask import" @("-c", "from app.main import app; print('Flask import OK')")

Write-Host "PASS: full local artefact pipeline completed."
