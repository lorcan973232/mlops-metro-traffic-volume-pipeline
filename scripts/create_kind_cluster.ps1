param(
    [string]$ClusterName = $(if ($env:KIND_CLUSTER_NAME) { $env:KIND_CLUSTER_NAME } else { "mlops-kind" }),
    [string]$NodeImage = $(if ($env:KIND_NODE_IMAGE) { $env:KIND_NODE_IMAGE } else { "kindest/node:v1.30.2" })
)

# Create or reuse the local Kind cluster used by deployment checks. The checks
# distinguish missing Docker/Kind setup from a project failure, which is helpful
# during a local demo on a Windows machine.
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

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "BLOCKED_BY_LOCAL_SETUP: docker is not installed. Install Docker Desktop: winget install Docker.DockerDesktop"
}
& docker info | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "BLOCKED_BY_LOCAL_SETUP: Docker daemon is not running. Start Docker Desktop and retry."
}
if (-not (Get-Command kind -ErrorAction SilentlyContinue)) {
    throw "BLOCKED_BY_LOCAL_SETUP: kind is not installed. Install Kind: winget install Kubernetes.kind"
}

$clusters = & kind get clusters
if ($clusters -contains $ClusterName) {
    Write-Host "Kind cluster already exists: $ClusterName"
} else {
    & kind create cluster --name $ClusterName --image $NodeImage
    if ($LASTEXITCODE -ne 0) {
        throw "Kind cluster creation failed."
    }
}

Write-Host "Available Kind clusters:"
& kind get clusters

if (Get-Command kubectl -ErrorAction SilentlyContinue) {
    & kubectl config set-cluster "kind-$ClusterName" --insecure-skip-tls-verify=true | Out-Null
    Write-Host "Current Kubernetes context:"
    & kubectl config current-context
    Write-Host "Cluster nodes:"
    & kubectl get nodes -o wide
} else {
    Write-Host "INFO: kubectl is not installed, so node status cannot be printed here."
}
