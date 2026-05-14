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
        mean_radius = 17.99
        mean_texture = 10.38
        mean_perimeter = 122.8
        mean_area = 1001.0
        mean_smoothness = 0.1184
        mean_compactness = 0.2776
        mean_concavity = 0.3001
        mean_concave_points = 0.1471
        mean_symmetry = 0.2419
        mean_fractal_dimension = 0.07871
        radius_error = 1.095
        texture_error = 0.9053
        perimeter_error = 8.589
        area_error = 153.4
        smoothness_error = 0.006399
        compactness_error = 0.04904
        concavity_error = 0.05373
        concave_points_error = 0.01587
        symmetry_error = 0.03003
        fractal_dimension_error = 0.006193
        worst_radius = 25.38
        worst_texture = 17.33
        worst_perimeter = 184.6
        worst_area = 2019.0
        worst_smoothness = 0.1622
        worst_compactness = 0.6656
        worst_concavity = 0.7119
        worst_concave_points = 0.2654
        worst_symmetry = 0.4601
        worst_fractal_dimension = 0.1189
    }
} | ConvertTo-Json -Depth 10

$prediction = Invoke-RestMethod `
    -Uri "$ApiUrl/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $payload `
    -TimeoutSec 10

if ($prediction.prediction -notin @("malignant", "benign")) {
    throw "Invalid prediction response: $($prediction | ConvertTo-Json -Depth 10)"
}
if (-not $prediction.model_version) {
    throw "Prediction response does not expose model_version: $($prediction | ConvertTo-Json -Depth 10)"
}
foreach ($label in @("malignant", "benign")) {
    if ($null -eq $prediction.probabilities.$label) {
        throw "Prediction response missing probability for $label"
    }
}
$prediction | ConvertTo-Json -Depth 10
