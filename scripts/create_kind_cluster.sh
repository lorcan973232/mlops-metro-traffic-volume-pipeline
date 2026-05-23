#!/usr/bin/env bash
set -euo pipefail

# Create or reuse the local Kind cluster used by the deployment workflow and demo.
# Keeping this separate from deploy_kind.sh lets setup checks prove the
# cluster step independently when Docker and Kind are available.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env_paths.sh
source "${SCRIPT_DIR}/env_paths.sh"

CLUSTER_NAME="${KIND_CLUSTER_NAME:-mlops-kind}"
KIND_NODE_IMAGE="${KIND_NODE_IMAGE:-kindest/node:v1.30.2}"

# Check Docker first because Kind runs Kubernetes nodes as Docker containers. If
# Docker is missing or stopped, the issue is local setup rather than project code.
if ! command -v docker >/dev/null 2>&1; then
  echo "BLOCKED_BY_LOCAL_SETUP: docker is not installed. Install Docker Desktop: https://docs.docker.com/desktop/"
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "BLOCKED_BY_LOCAL_SETUP: Docker daemon is not running. Start Docker Desktop and retry."
  exit 1
fi

if ! command -v kind >/dev/null 2>&1; then
  echo "BLOCKED_BY_LOCAL_SETUP: kind is not installed. Install Kind: https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
  exit 1
fi

# Reuse an existing cluster during a demo so rerunning the script does not
# destroy a working environment. Creating it only when missing keeps the command
# safe to run before local verification or a GitHub-style smoke test.
if kind get clusters | grep -qx "${CLUSTER_NAME}"; then
  echo "Kind cluster already exists: ${CLUSTER_NAME}"
else
  kind create cluster --name "${CLUSTER_NAME}" --image "${KIND_NODE_IMAGE}"
fi

echo "Available Kind clusters:"
kind get clusters

# kubectl output is helpful context, but kubectl may be installed separately
# from Kind. The script reports that clearly instead of failing after the cluster
# has already been created.
if command -v kubectl >/dev/null 2>&1; then
  kubectl config set-cluster "kind-${CLUSTER_NAME}" --insecure-skip-tls-verify=true >/dev/null 2>&1 || true
  echo "Current Kubernetes context:"
  kubectl config current-context || true
  echo "Cluster nodes:"
  kubectl get nodes -o wide || true
else
  echo "INFO: kubectl is not installed, so node status cannot be printed here."
fi
