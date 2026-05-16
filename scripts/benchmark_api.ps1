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

Write-Host "Benchmarking API at $ApiUrl" -ForegroundColor Green

# Create output directory if it doesn't exist
$OutputDir = Split-Path -Parent $Output
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# Run benchmark
python scripts/benchmark_api.py $ApiUrl --samples $Samples --warmup $Warmup --output $Output

if ($LASTEXITCODE -ne 0) {
    Write-Host "Benchmark failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Benchmark report saved to: $Output" -ForegroundColor Green
