[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$ApiUrl,

    [Parameter(Mandatory=$false)]
    [int]$Samples = 100,

    [Parameter(Mandatory=$false)]
    [int]$Warmup = 10,

    [Parameter(Mandatory=$false)]
    [string]$Output = "reports/benchmarks/api_sla_report.json"
)

# Windows wrapper for the API latency benchmark. It is used after Flask, Docker,
# or Kind is already serving the model, then writes the same JSON SLA report as
# the Bash/Python path under reports/benchmarks/.
Write-Host "Benchmarking API at $ApiUrl" -ForegroundColor Green

# The report directory is created here so a fresh checkout can produce benchmark
# evidence without manual folder setup.
$OutputDir = Split-Path -Parent $Output
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# The Python script sends real `/predict` requests using the shared example
# payload, so the benchmark checks the served model path rather than a mock.
python scripts/benchmark_api.py $ApiUrl --samples $Samples --warmup $Warmup --output $Output

if ($LASTEXITCODE -ne 0) {
    Write-Host "Benchmark failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Benchmark report saved to: $Output" -ForegroundColor Green
