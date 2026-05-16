from __future__ import annotations

from pathlib import Path

import yaml

EXPECTED_WORKFLOWS = {
    "ci.yml",
    "data-preprocessing.yml",
    "train-and-evaluate.yml",
    "continuous-training.yml",
    "docker-build.yml",
    "deploy.yml",
    "monitoring.yml",
}

REQUIRED_COMMAND_PATHS = [
    "scripts/check_setup.sh",
    "scripts/smoke_test_api.sh",
    "scripts/monitor.py",
    "scripts/check_drift.py",
    "deployment/kind/",
]

WINDOWS_SCRIPT_PATHS = [
    "scripts/setup_local.ps1",
    "scripts/check_setup.ps1",
    "scripts/smoke_test_api.ps1",
    "scripts/create_kind_cluster.ps1",
    "scripts/deploy_kind.ps1",
]


def load_workflow(name: str) -> dict:
    return yaml.safe_load((Path(".github/workflows") / name).read_text(encoding="utf-8"))


def test_required_github_actions_workflows_are_present_and_valid_yaml() -> None:
    workflow_dir = Path(".github/workflows")
    actual = {path.name for path in workflow_dir.glob("*.yml")}
    assert EXPECTED_WORKFLOWS.issubset(actual)
    for workflow_name in EXPECTED_WORKFLOWS:
        workflow = load_workflow(workflow_name)
        assert workflow["jobs"]
        assert workflow["name"]
        assert workflow["permissions"] == {"contents": "read"}


def test_workflow_triggers_and_dependencies_show_lifecycle() -> None:
    ci = load_workflow("ci.yml")
    assert ci["on"]["push"]["branches"] == ["main", "develop"]
    assert ci["on"]["pull_request"]["branches"] == ["main", "develop"]
    assert ci["jobs"]["quality-gates"]["needs"] == "setup-check"

    data = load_workflow("data-preprocessing.yml")
    assert data["jobs"]["preprocess"]["needs"] == "ingest"

    train = load_workflow("train-and-evaluate.yml")
    assert train["jobs"]["train-evaluate"]["needs"] == "prepare-data"

    ct = load_workflow("continuous-training.yml")
    assert "schedule" in ct["on"]
    assert ct["jobs"]["quality-gate"]["needs"] == "retrain-candidate"
    assert ct["jobs"]["promote-model"]["needs"] == "quality-gate"
    ct_text = Path(".github/workflows/continuous-training.yml").read_text(encoding="utf-8")
    assert "balanced_accuracy_above_minimum" in ct_text
    assert "raise SystemExit(\"Continuous Training rejected candidate model.\")" in ct_text

    monitoring = load_workflow("monitoring.yml")
    assert "schedule" in monitoring["on"]
    assert monitoring["jobs"]["batch-monitoring"]["needs"] == "prepare-model-metadata"
    monitoring_text = Path(".github/workflows/monitoring.yml").read_text(encoding="utf-8")
    assert "missing_monitoring_fields" in monitoring_text
    assert "missing_drift_fields" in monitoring_text


def test_deploy_workflow_and_kind_manifests_are_kind_only() -> None:
    relevant_paths = [
        Path(".github/workflows/deploy.yml"),
        Path("deployment/kind/deployment.yaml"),
        Path("deployment/kind/service.yaml"),
        Path("deployment/kind/README.md"),
    ]
    content = "\n".join(path.read_text(encoding="utf-8").lower() for path in relevant_paths)
    assert "kind create cluster" in content
    assert "kind load docker-image" in content
    assert "mlops-kind" in content
    assert "kubectl apply -f deployment/kind/" in content
    disallowed_cloud_terms = ["g" + "oogle", "g" + "cloud"]
    assert not any(term in content for term in disallowed_cloud_terms)


def test_workflows_reference_existing_commands_and_upload_artifacts() -> None:
    workflow_text = "\n".join(
        path.read_text(encoding="utf-8") for path in Path(".github/workflows").glob("*.yml")
    )
    for required_path in REQUIRED_COMMAND_PATHS:
        assert Path(required_path).exists()
        assert required_path in workflow_text

    for windows_script in WINDOWS_SCRIPT_PATHS:
        assert Path(windows_script).exists()

    assert "python -m src.data" in workflow_text
    assert "python -m src.preprocess" in workflow_text
    assert "python -m src.train" in workflow_text
    assert "python -m src.evaluate" in workflow_text
    assert "data/processed/winequality-red-processed.csv" in workflow_text
    assert "en" + "ergy-efficiency-processed.csv" not in workflow_text
    assert "actions/checkout@v6.0.2" in workflow_text
    assert "actions/setup-python@v6.2.0" in workflow_text
    assert "actions/upload-artifact@v7.0.1" in workflow_text
    assert "actions/download-artifact@v8.0.1" in workflow_text
    assert "actions/checkout@v4" not in workflow_text
    assert "actions/setup-python@v5" not in workflow_text
    assert "actions/upload-artifact@v4" not in workflow_text
    assert "actions/download-artifact@v4" not in workflow_text
    assert "latest_metrics.json" in workflow_text
    assert "quality_gate_report.json" in workflow_text
    assert "model_metadata.json" in workflow_text
    assert "confusion_matrix_normalized.json" in workflow_text
    assert "cross_validation_results.json" in workflow_text
    assert "mlops-flask-api:${{ github.sha }}" in workflow_text
    assert "python -c \"from app.main import app; print('Flask import OK')\"" in workflow_text
    assert "kubectl get svc -o wide" in workflow_text


def test_workflows_do_not_hardcode_credentials() -> None:
    workflow_text = "\n".join(
        path.read_text(encoding="utf-8") for path in Path(".github/workflows").glob("*.yml")
    ).lower()
    forbidden_fragments = [
        "gh" + "p_",
        "github" + "_pat_",
        "pass" + "word=",
        "sec" + "ret=",
        "tok" + "en=",
        "begin rsa" + " private key",
        "begin openssh" + " private key",
    ]
    assert not any(fragment in workflow_text for fragment in forbidden_fragments)


def test_docker_workflow_has_verified_dataset_build_context() -> None:
    raw_dataset = Path("data/raw/winequality-red.csv")
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert raw_dataset.exists()
    assert "COPY data/raw/ data/raw/" in dockerfile
    assert "data/raw/*.csv" not in dockerignore
    assert "!data/raw/winequality-red.csv" in gitignore


def test_smoke_test_uses_valid_prediction_feature_names() -> None:
    smoke_script = Path("scripts/smoke_test_api.sh").read_text(encoding="utf-8")
    for feature_name in [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "ph",
        "sulphates",
        "alcohol",
    ]:
        assert feature_name in smoke_script


def test_monitoring_workflow_and_scripts_emit_required_model_management_fields() -> None:
    workflow = Path(".github/workflows/monitoring.yml").read_text(encoding="utf-8")
    monitor_script = Path("scripts/monitor.py").read_text(encoding="utf-8")
    drift_script = Path("scripts/check_drift.py").read_text(encoding="utf-8")
    registry = Path("src/model_registry.py").read_text(encoding="utf-8")

    assert "python scripts/monitor.py" in workflow
    assert "python scripts/check_drift.py" in workflow
    assert "reports/monitoring/" in workflow
    assert "model_metadata.json" in workflow
    assert "--api-url" in monitor_script
    assert "api_aware_monitoring" in monitor_script
    assert "offline_simulated_monitoring" in monitor_script
    assert "retraining_required" in monitor_script
    assert "retraining_required" in drift_script
    assert "drift_score" in drift_script
    assert "model_registry_summary" in registry


def test_quality_gate_includes_balanced_accuracy_contract() -> None:
    evaluate = Path("src/evaluate.py").read_text(encoding="utf-8")
    assert "MIN_BALANCED_ACCURACY" in evaluate
    assert "balanced_accuracy_above_minimum" in evaluate
    assert "candidate_balanced_accuracy" in evaluate
