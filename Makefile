PYTHON ?= python
ifeq ($(OS),Windows_NT)
# On Windows, Git Bash is used only for the Bash route. PowerShell targets below
# give the normal Windows path, which avoids relying on WSL during marking.
BASH ?= "C:/Program Files/Git/bin/bash.exe"
else
BASH ?= bash
endif
IMAGE_NAME ?= mlops-flask-api:latest
KIND_CLUSTER_NAME ?= mlops-kind
API_URL ?= http://127.0.0.1:8080

# The Makefile is a convenience layer over the same scripts and Python modules
# documented in the README. It does not add a separate workflow path; it gives
# the student and marker short commands for setup, tests, model evidence, Docker,
# Kind, monitoring, and security checks.

.PHONY: setup setup-ps check-setup check-setup-ps test lint workflow-test data preprocess model-select train evaluate predict run-api flask-import api-smoke api-smoke-ps docker-build docker-run kind-create kind-create-ps kind-load kind-deploy kind-deploy-ps kind-smoke-test kind-smoke-test-ps deployment-readiness monitor monitor-api drift-check security-scan workflow-validate full-local-verify

setup:
	# Bash setup is for Linux, Git Bash, and GitHub Actions routes.
	$(BASH) scripts/setup_local.sh

setup-ps:
	# PowerShell setup is the recommended Windows route.
	powershell -ExecutionPolicy Bypass -File scripts/setup_local.ps1

check-setup:
	# Local checks separate missing tools from project failures.
	$(BASH) scripts/check_setup.sh

check-setup-ps:
	powershell -ExecutionPolicy Bypass -File scripts/check_setup.ps1

test:
	# Compile and pytest protect the Python, API, and evidence contracts.
	$(PYTHON) -m compileall app src tests
	$(PYTHON) -m pytest -q

lint:
	# Ruff catches simple Python quality issues before CI does.
	$(PYTHON) -m ruff check src tests

workflow-test:
	# Workflow tests check that Actions files still point at real scripts and
	# upload evidence. This catches broken YAML links before a remote run fails.
	$(PYTHON) -m pytest tests/test_workflows.py -q

data:
	# Ingest and validate the fixed UCI source file.
	$(PYTHON) -m src.data

preprocess:
	# Write the deterministic processed CSV used by training and monitoring.
	$(PYTHON) -m src.preprocess

model-select:
	# Compare baseline, tuned, and ensemble candidates before training.
	$(PYTHON) -m src.model_selection

train:
	# Save the model bundle consumed by Flask, Docker, and Kind.
	$(PYTHON) -m src.train

evaluate:
	# Write metrics and enforce the quality gate before registration.
	$(PYTHON) -m src.evaluate
	$(PYTHON) -m src.model_registry

predict:
	# Prediction is a local smoke path for the saved model and shared schema.
	$(PYTHON) -m src.predict

run-api:
	# Start the same Flask app used by Docker and Kind so the browser demo can
	# call `/health` and `/predict` locally.
	$(PYTHON) -m app.main

flask-import:
	# This quick import check catches model-loading or Flask app wiring issues
	# early in CI before longer Docker or Kind jobs run.
	$(PYTHON) -c "from app.main import app; print('Flask import OK')"

api-smoke:
	# The smoke test calls `/health` and `/predict`, so it checks both serving and
	# the saved model contract rather than only checking that a port is open.
	$(BASH) scripts/smoke_test_api.sh $(API_URL)

api-smoke-ps:
	powershell -ExecutionPolicy Bypass -File scripts/smoke_test_api.ps1 -ApiUrl $(API_URL)

docker-build:
	# Build the image that is later loaded into Kind. The Dockerfile reruns core
	# pipeline checks so the container is tied to the same training evidence.
	docker build -t $(IMAGE_NAME) .

docker-run:
	# Run the container on 5001 so it does not clash with a local Flask process on
	# 5000 during a live demo.
	docker run --rm -p 5001:5000 $(IMAGE_NAME)

kind-create:
	KIND_CLUSTER_NAME=$(KIND_CLUSTER_NAME) $(BASH) scripts/create_kind_cluster.sh

kind-create-ps:
	powershell -ExecutionPolicy Bypass -File scripts/create_kind_cluster.ps1 -ClusterName $(KIND_CLUSTER_NAME)

kind-load:
	# Kind uses a local Docker image, not a registry pull. Loading the image keeps
	# the Kubernetes demo repeatable without cloud credentials.
	kind load docker-image $(IMAGE_NAME) --name $(KIND_CLUSTER_NAME)

kind-deploy:
	IMAGE_NAME=$(IMAGE_NAME) KIND_CLUSTER_NAME=$(KIND_CLUSTER_NAME) $(BASH) scripts/deploy_kind.sh

kind-deploy-ps:
	powershell -ExecutionPolicy Bypass -File scripts/deploy_kind.ps1 -ClusterName $(KIND_CLUSTER_NAME) -ImageName $(IMAGE_NAME)

kind-smoke-test:
	$(BASH) scripts/smoke_test_api.sh $(API_URL)

kind-smoke-test-ps:
	powershell -ExecutionPolicy Bypass -File scripts/smoke_test_api.ps1 -ApiUrl $(API_URL)

deployment-readiness:
	# This helper separates missing local Docker/Kind tools from real project
	# failures, which is useful on a machine that cannot run Kubernetes locally.
	$(PYTHON) scripts/check_deployment_readiness.py

monitor:
	# Offline monitoring checks the processed dataset schema and drift reports.
	$(PYTHON) scripts/monitor.py

monitor-api:
	# API-aware monitoring is used after Flask, Docker, or Kind is running.
	$(PYTHON) scripts/monitor.py --api-url $(API_URL)

drift-check:
	# The drift check writes retraining flags without claiming live production
	# monitoring. It is a repeatable coursework monitoring stage.
	$(PYTHON) scripts/check_drift.py

security-scan:
	# The security scan creates a snapshot of dependency, secret, Docker, and SBOM
	# evidence. It does not claim future packages will stay safe.
	$(PYTHON) scripts/security_scan.py

workflow-validate:
	$(PYTHON) - <<'PY'
	from pathlib import Path
	import yaml
	for path in sorted(Path(".github/workflows").glob("*.yml")):
	    yaml.safe_load(path.read_text(encoding="utf-8"))
	    print(f"PASS: {path}")
	PY

# This target is the quickest marker-facing local check. Kind is kept separate
# because cluster creation depends on the machine's Docker and Kubernetes setup.
full-local-verify: check-setup test lint workflow-test workflow-validate data preprocess model-select train evaluate predict monitor drift-check flask-import security-scan docker-build
	@echo "PASS: full local artefact verification completed. Run kind-deploy and kind-smoke-test for the live Kind deployment path."
