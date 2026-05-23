param(
    [string]$ApiUrl = "http://127.0.0.1:8080"
)

# This PowerShell smoke test is the Windows equivalent of `smoke_test_api.sh`.
# It checks the deployed service has loaded the model through `/health`, then
# sends a real 15-feature traffic payload to `/predict` and checks the response
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
if ($health.feature_count -ne 15) {
    throw "Health response feature_count mismatch: $($health | ConvertTo-Json -Depth 10)"
}
$health | ConvertTo-Json -Depth 10

# Use the same feature values as the browser example. This links the smoke test,
# UI demo, and model schema to one shared prediction contract.
$payload = @{
    features = @{
        temp = 288.28
        rain_1h = 0.0
        snow_1h = 0.0
        clouds_all = 40.0
        hour = 17
        month = 10
        day_of_week = 2
        is_weekend = 0
        is_holiday = 0
        weather_main = "Clouds"
        lag_1h_volume = 5545.0
        lag_24h_volume = 6015.0
        lag_168h_volume = 5365.0
        rolling_3h_volume = 5480.0
        rolling_24h_volume = 4210.0
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
if ($prediction.prediction_label -notin @("normal traffic", "high traffic")) {
    throw "Prediction label mismatch: $($prediction | ConvertTo-Json -Depth 10)"
}
if (-not $prediction.model_version) {
    throw "Prediction response does not expose model_version: $($prediction | ConvertTo-Json -Depth 10)"
}
if ($prediction.target -ne "high_traffic") {
    throw "Prediction response target mismatch: $($prediction | ConvertTo-Json -Depth 10)"
}
if ($null -eq $prediction.confidence) {
    throw "Prediction response does not expose confidence: $($prediction | ConvertTo-Json -Depth 10)"
}
$prediction | ConvertTo-Json -Depth 10
