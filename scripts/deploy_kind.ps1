param(
    [string]$ClusterName = $(if ($env:KIND_CLUSTER_NAME) { $env:KIND_CLUSTER_NAME } else { "mlops-kind" }),
    [string]$ImageName = $(if ($env:IMAGE_NAME) { $env:IMAGE_NAME } else { "mlops-flask-api:latest" }),
    [string]$NodeImage = $(if ($env:KIND_NODE_IMAGE) { $env:KIND_NODE_IMAGE } else { "kindest/node:v1.30.2" }),
    [switch]$StartPortForward
)

# This is the Windows Kind deployment route. It builds the local Docker image,
# creates or reuses the Kind cluster, loads that exact image into the cluster,
# applies the manifests, and prints the port-forward/smoke-test commands needed
# for a local demo.
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$wingetLinks = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links"
if ((Test-Path $wingetLinks) -and ($env:PATH -notlike "*$wingetLinks*")) {
    $env:PATH = "$env:PATH;$wingetLinks"
}
$dockerBin = "C:\Program Files\Docker\Docker\resources\bin"
if ((Test-Path $dockerBin) -and ($env:PATH -notlike "*$dockerBin*")) {
    $env:PATH = "$env:PATH;$dockerBin"
}

if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
    throw "BLOCKED_BY_LOCAL_SETUP: kubectl is not installed. Install kubectl: winget install Kubernetes.kubectl"
}

Write-Host "Building Docker image: $ImageName"
& docker build -t $ImageName .
if ($LASTEXITCODE -ne 0) {
    throw "Docker build failed."
}

Write-Host "Creating or reusing Kind cluster: $ClusterName"
& powershell -ExecutionPolicy Bypass -File scripts/create_kind_cluster.ps1 -ClusterName $ClusterName -NodeImage $NodeImage
if ($LASTEXITCODE -ne 0) {
    throw "Kind cluster setup failed."
}

Write-Host "Loading Docker image into Kind: $ImageName"
# Kind uses a local image load rather than a registry pull, which keeps the demo
# reproducible without cloud credentials or image-publishing steps.
& kind load docker-image $ImageName --name $ClusterName
if ($LASTEXITCODE -ne 0) {
    throw "kind load docker-image failed."
}

Write-Host "Applying Kind manifests from deployment/kind/"
& kubectl apply -f deployment/kind/
if ($LASTEXITCODE -ne 0) {
    throw "kubectl apply failed."
}

Write-Host "Restarting deployment so Kind uses the freshly loaded local image"
& kubectl rollout restart deployment/mlops-flask-api
if ($LASTEXITCODE -ne 0) {
    throw "Kubernetes rollout restart failed."
}

Write-Host "Waiting for rollout"
& kubectl rollout status deployment/mlops-flask-api --timeout=360s
if ($LASTEXITCODE -ne 0) {
    throw "Kubernetes rollout failed."
}

Write-Host "Deployment resources:"
& kubectl get all -l app=mlops-flask-api -o wide

Write-Host "Run a local service tunnel and smoke test:"
Write-Host "  kubectl port-forward service/mlops-flask-api 8080:80"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/smoke_test_api.ps1 -ApiUrl http://127.0.0.1:8080"

if ($StartPortForward) {
    Write-Host "Starting background port-forward on http://127.0.0.1:8080"
    Start-Process `
        -FilePath "kubectl" `
        -ArgumentList @("port-forward", "service/mlops-flask-api", "8080:80") `
        -WindowStyle Hidden
}
