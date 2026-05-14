param(
    [string]$ApiUrl = "http://127.0.0.1:8080"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$health = Invoke-RestMethod -Uri "$ApiUrl/health" -Method Get -TimeoutSec 10
if ($health.status -ne "healthy" -or $health.model_loaded -ne $true) {
    throw "Invalid health response: $($health | ConvertTo-Json -Depth 10)"
}
if (-not $health.model_version) {
    throw "Health response does not expose model_version: $($health | ConvertTo-Json -Depth 10)"
}
$health | ConvertTo-Json -Depth 10

$payload = @{
    features = @{
        fixed_acidity = 7.0
        volatile_acidity = 0.27
        citric_acid = 0.36
        residual_sugar = 20.7
        chlorides = 0.045
        free_sulfur_dioxide = 45.0
        total_sulfur_dioxide = 170.0
        density = 1.001
        pH = 3.0
        sulphates = 0.45
        alcohol = 8.8
    }
} | ConvertTo-Json -Depth 10

$prediction = Invoke-RestMethod `
    -Uri "$ApiUrl/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $payload `
    -TimeoutSec 10

if ($prediction.prediction -notin @("low", "medium", "high")) {
    throw "Invalid prediction response: $($prediction | ConvertTo-Json -Depth 10)"
}
if (-not $prediction.model_version) {
    throw "Prediction response does not expose model_version: $($prediction | ConvertTo-Json -Depth 10)"
}
foreach ($label in @("low", "medium", "high")) {
    if ($null -eq $prediction.probabilities.$label) {
        throw "Prediction response missing probability for $label"
    }
}
$prediction | ConvertTo-Json -Depth 10
