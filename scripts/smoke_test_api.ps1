param(
    [string]$ApiUrl = "http://127.0.0.1:8080"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$health = Invoke-RestMethod -Uri "$ApiUrl/health" -Method Get -TimeoutSec 10
if ($health.status -ne "healthy" -or $health.model_loaded -ne $true) {
    throw "Invalid health response: $($health | ConvertTo-Json -Depth 10)"
}
if ($health.task_type -ne "regression") {
    throw "Health response does not expose regression task type: $($health | ConvertTo-Json -Depth 10)"
}
if (-not $health.model_version) {
    throw "Health response does not expose model_version: $($health | ConvertTo-Json -Depth 10)"
}
$health | ConvertTo-Json -Depth 10

$payload = @{
    features = @{
        relative_compactness = 0.76
        surface_area = 661.5
        wall_area = 416.5
        roof_area = 122.5
        overall_height = 7.0
        orientation = 2
        glazing_area = 0.4
        glazing_area_distribution = 5
    }
} | ConvertTo-Json -Depth 10

$prediction = Invoke-RestMethod `
    -Uri "$ApiUrl/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $payload `
    -TimeoutSec 10

if ($null -eq $prediction.prediction -or [double]$prediction.prediction -le 0) {
    throw "Invalid regression prediction response: $($prediction | ConvertTo-Json -Depth 10)"
}
if (-not $prediction.model_version) {
    throw "Prediction response does not expose model_version: $($prediction | ConvertTo-Json -Depth 10)"
}
if ($prediction.target -ne "heating_load") {
    throw "Prediction response target mismatch: $($prediction | ConvertTo-Json -Depth 10)"
}
$prediction | ConvertTo-Json -Depth 10
