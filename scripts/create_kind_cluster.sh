#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="${KIND_CLUSTER_NAME:-mlops-kind}"
KIND_NODE_IMAGE="${KIND_NODE_IMAGE:-kindest/node:v1.30.2}"

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

if kind get clusters | grep -qx "${CLUSTER_NAME}"; then
  echo "Kind cluster already exists: ${CLUSTER_NAME}"
else
  kind create cluster --name "${CLUSTER_NAME}" --image "${KIND_NODE_IMAGE}"
fi

echo "Available Kind clusters:"
kind get clusters

if command -v kubectl >/dev/null 2>&1; then
  kubectl config set-cluster "kind-${CLUSTER_NAME}" --insecure-skip-tls-verify=true >/dev/null 2>&1 || true
  echo "Current Kubernetes context:"
  kubectl config current-context || true
  echo "Cluster nodes:"
  kubectl get nodes -o wide || true
else
  echo "INFO: kubectl is not installed, so node status cannot be printed here."
fi
