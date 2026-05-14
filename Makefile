PYTHON ?= python
BASH ?= bash
IMAGE_NAME ?= mlops-flask-api:latest
KIND_CLUSTER_NAME ?= mlops-kind
API_URL ?= http://127.0.0.1:8080

.PHONY: setup setup-ps check-setup check-setup-ps test data preprocess model-select train evaluate run-api docker-build docker-run kind-create kind-create-ps kind-load kind-deploy kind-deploy-ps kind-smoke-test kind-smoke-test-ps monitor drift-check full-local-verify

setup:
	$(BASH) scripts/setup_local.sh

setup-ps:
	powershell -ExecutionPolicy Bypass -File scripts/setup_local.ps1

check-setup:
	$(BASH) scripts/check_setup.sh

check-setup-ps:
	powershell -ExecutionPolicy Bypass -File scripts/check_setup.ps1

test:
	$(PYTHON) -m compileall app src tests
	$(PYTHON) -m pytest

data:
	$(PYTHON) -m src.data

preprocess:
	$(PYTHON) -m src.preprocess

model-select:
	$(PYTHON) -m src.model_selection

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate
	$(PYTHON) -m src.model_registry

run-api:
	$(PYTHON) -m app.main

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm -p 8080:8080 $(IMAGE_NAME)

kind-create:
	KIND_CLUSTER_NAME=$(KIND_CLUSTER_NAME) $(BASH) scripts/create_kind_cluster.sh

kind-create-ps:
	powershell -ExecutionPolicy Bypass -File scripts/create_kind_cluster.ps1 -ClusterName $(KIND_CLUSTER_NAME)

kind-load:
	kind load docker-image $(IMAGE_NAME) --name $(KIND_CLUSTER_NAME)

kind-deploy:
	IMAGE_NAME=$(IMAGE_NAME) KIND_CLUSTER_NAME=$(KIND_CLUSTER_NAME) $(BASH) scripts/deploy_kind.sh

kind-deploy-ps:
	powershell -ExecutionPolicy Bypass -File scripts/deploy_kind.ps1 -ClusterName $(KIND_CLUSTER_NAME) -ImageName $(IMAGE_NAME)

kind-smoke-test:
	$(BASH) scripts/smoke_test_api.sh $(API_URL)

kind-smoke-test-ps:
	powershell -ExecutionPolicy Bypass -File scripts/smoke_test_api.ps1 -ApiUrl $(API_URL)

monitor:
	$(PYTHON) scripts/monitor.py

drift-check:
	$(PYTHON) scripts/check_drift.py

full-local-verify: check-setup test data preprocess model-select train evaluate monitor drift-check docker-build
	@echo "Run kind-deploy and kind-smoke-test after Docker, Kind, and kubectl are confirmed available."
