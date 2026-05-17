#!/usr/bin/env bash
set -euo pipefail

# This script is the local Linux/Git Bash version of the Kind deployment path. It
# builds the Docker image, loads it into Kind, applies the manifests, and then
# tells the user exactly how to port-forward and smoke-test the API.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env_paths.sh
source "${SCRIPT_DIR}/env_paths.sh"

CLUSTER_NAME="${KIND_CLUSTER_NAME:-mlops-kind}"
IMAGE_NAME="${IMAGE_NAME:-mlops-flask-api:latest}"
API_URL="${API_URL:-http://127.0.0.1:8080}"

# kubectl is needed for every step after image loading. Failing here gives a
# clear setup message before Docker spends time building an image.
if ! command -v kubectl >/dev/null 2>&1; then
  echo "BLOCKED_BY_LOCAL_SETUP: kubectl is not installed. Install kubectl: https://kubernetes.io/docs/tasks/tools/"
  exit 1
fi

# Build from the repository root so the image includes the app, saved model,
# reports, and data paths that the Dockerfile expects.
echo "Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" .

# Kind does not pull this image from a registry. Loading the local image keeps the
# coursework deployment reproducible without needing cloud credentials.
echo "Creating or reusing Kind cluster: ${CLUSTER_NAME}"
KIND_CLUSTER_NAME="${CLUSTER_NAME}" bash scripts/create_kind_cluster.sh

echo "Loading Docker image into Kind: ${IMAGE_NAME}"
kind load docker-image "${IMAGE_NAME}" --name "${CLUSTER_NAME}"

echo "Applying Kind manifests from deployment/kind/"
kubectl apply -f deployment/kind/

# The deployment may already exist from a previous demo run. Restarting it makes
# Kubernetes pick up the freshly loaded local image before the smoke test.
echo "Restarting deployment so Kind uses the freshly loaded local image"
kubectl rollout restart deployment/mlops-flask-api

echo "Waiting for rollout"
kubectl rollout status deployment/mlops-flask-api --timeout=360s

echo "Deployment resources:"
kubectl get all -l app=mlops-flask-api -o wide

cat <<'NEXT'
Run a local service tunnel and smoke test:
  kubectl port-forward service/mlops-flask-api 8080:80
  scripts/smoke_test_api.sh http://127.0.0.1:8080
NEXT

# The optional background port-forward is useful for automation, while the
# default printed command is safer for a manual live demo.
if [[ "${START_PORT_FORWARD:-0}" == "1" ]]; then
  echo "Starting background port-forward for ${API_URL}"
  kubectl port-forward service/mlops-flask-api 8080:80 >/tmp/mlops-flask-api-port-forward.log 2>&1 &
  echo "Port-forward PID: $!"
fi
