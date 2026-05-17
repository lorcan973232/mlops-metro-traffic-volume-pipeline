param(
    [string]$ApiUrl = "http://127.0.0.1:8080"
)

# This PowerShell smoke test is the Windows equivalent of `smoke_test_api.sh`.
# It proves the deployed service has loaded the model through `/health`, then
# sends a real 11-feature wine payload to `/predict` and checks the response
# fields used by the browser UI and workflow logs.
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$health = Invoke-RestMethod -Uri "$ApiUrl/health" -Method Get -TimeoutSec 10
# `/health` must prove that the model is loaded, not just that Flask is running.
# Without this, Docker or Kind could appear healthy while prediction would fail.
if ($health.status -ne "healthy" -or $health.model_loaded -ne $true) {
    throw "Invalid health response: $($health | ConvertTo-Json -Depth 10)"
}
if ($health.task_type -ne "classification") {
    throw "Health response does not expose classification task type: $($health | ConvertTo-Json -Depth 10)"
}
if (-not $health.model_version) {
    throw "Health response does not expose model_version: $($health | ConvertTo-Json -Depth 10)"
}
if ($health.feature_count -ne 11) {
    throw "Health response feature_count mismatch: $($health | ConvertTo-Json -Depth 10)"
}
$health | ConvertTo-Json -Depth 10

# Use the same feature values as the browser example. This links the smoke test,
# UI demo, and model schema to one shared prediction contract.
$payload = @{
    features = @{
        fixed_acidity = 7.4
        volatile_acidity = 0.7
        citric_acid = 0.0
        residual_sugar = 1.9
        chlorides = 0.076
        free_sulfur_dioxide = 11.0
        total_sulfur_dioxide = 34.0
        density = 0.9978
        ph = 3.51
        sulphates = 0.56
        alcohol = 9.4
    }
} | ConvertTo-Json -Depth 10

$prediction = Invoke-RestMethod `
    -Uri "$ApiUrl/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $payload `
    -TimeoutSec 10

# The response checks guard against partial API success. A 200 response is not
# enough if it does not include the label, confidence, target, and model version
# expected by the UI and monitoring scripts.
if ($prediction.prediction -notin @(0, 1)) {
    throw "Invalid classification prediction response: $($prediction | ConvertTo-Json -Depth 10)"
}
if ($prediction.prediction_label -notin @("standard quality", "good quality")) {
    throw "Prediction label mismatch: $($prediction | ConvertTo-Json -Depth 10)"
}
if (-not $prediction.model_version) {
    throw "Prediction response does not expose model_version: $($prediction | ConvertTo-Json -Depth 10)"
}
if ($prediction.target -ne "quality_label") {
    throw "Prediction response target mismatch: $($prediction | ConvertTo-Json -Depth 10)"
}
if ($null -eq $prediction.confidence) {
    throw "Prediction response does not expose confidence: $($prediction | ConvertTo-Json -Depth 10)"
}
$prediction | ConvertTo-Json -Depth 10
