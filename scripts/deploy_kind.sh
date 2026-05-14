#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env_paths.sh
source "${SCRIPT_DIR}/env_paths.sh"

CLUSTER_NAME="${KIND_CLUSTER_NAME:-mlops-kind}"
IMAGE_NAME="${IMAGE_NAME:-mlops-flask-api:latest}"
API_URL="${API_URL:-http://127.0.0.1:8080}"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "BLOCKED_BY_LOCAL_SETUP: kubectl is not installed. Install kubectl: https://kubernetes.io/docs/tasks/tools/"
  exit 1
fi

echo "Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" .

echo "Creating or reusing Kind cluster: ${CLUSTER_NAME}"
KIND_CLUSTER_NAME="${CLUSTER_NAME}" bash scripts/create_kind_cluster.sh

echo "Loading Docker image into Kind: ${IMAGE_NAME}"
kind load docker-image "${IMAGE_NAME}" --name "${CLUSTER_NAME}"

echo "Applying Kind manifests from deployment/kind/"
kubectl apply -f deployment/kind/

echo "Restarting deployment so Kind uses the freshly loaded local image"
kubectl rollout restart deployment/mlops-flask-api

echo "Waiting for rollout"
kubectl rollout status deployment/mlops-flask-api --timeout=180s

echo "Deployment resources:"
kubectl get all -l app=mlops-flask-api -o wide

cat <<'NEXT'
Run a local service tunnel and smoke test:
  kubectl port-forward service/mlops-flask-api 8080:80
  scripts/smoke_test_api.sh http://127.0.0.1:8080
NEXT

if [[ "${START_PORT_FORWARD:-0}" == "1" ]]; then
  echo "Starting background port-forward for ${API_URL}"
  kubectl port-forward service/mlops-flask-api 8080:80 >/tmp/mlops-flask-api-port-forward.log 2>&1 &
  echo "Port-forward PID: $!"
fi
