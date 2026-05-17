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
    "model-analysis.yml",
    "repository-visibility-check.yml",
    "security-scan.yml",
    "bash-script-verification.yml",
}

REQUIRED_COMMAND_PATHS = [
    "scripts/check_setup.sh",
    "scripts/smoke_test_api.sh",
    "scripts/monitor.py",
    "scripts/check_drift.py",
    "scripts/explain_model.py",
    "scripts/fairness_audit.py",
    "scripts/cost_benefit_analysis.py",
    "deployment/kind/",
    "scripts/security_scan.py",
    "scripts/check_repo_visibility.py",
    "scripts/final_readiness_check.py",
    "scripts/check_bash_environment.sh",
]

WINDOWS_SCRIPT_PATHS = [
    "scripts/setup_local.ps1",
    "scripts/check_setup.ps1",
    "scripts/run_pipeline.ps1",
    "scripts/smoke_test_api.ps1",
    "scripts/create_kind_cluster.ps1",
    "scripts/deploy_kind.ps1",
    "scripts/check_bash_environment.ps1",
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

    model_analysis = load_workflow("model-analysis.yml")
    assert model_analysis["jobs"]["tier3-analysis"]["env"]["FAST_MODE"] == "1"
    model_analysis_text = Path(".github/workflows/model-analysis.yml").read_text(
        encoding="utf-8"
    )
    assert "python scripts/explain_model.py" in model_analysis_text
    assert "python scripts/fairness_audit.py" in model_analysis_text
    assert "python scripts/cost_benefit_analysis.py" in model_analysis_text
    assert "tier3-model-analysis-reports" in model_analysis_text

    visibility = load_workflow("repository-visibility-check.yml")
    assert "schedule" in visibility["on"]
    visibility_text = Path(".github/workflows/repository-visibility-check.yml").read_text(
        encoding="utf-8"
    )
    assert "python scripts/check_repo_visibility.py" in visibility_text
    assert "repository-visibility-evidence" in visibility_text

    bash_verification = load_workflow("bash-script-verification.yml")
    assert bash_verification["jobs"]["bash-verification"]
    bash_text = Path(".github/workflows/bash-script-verification.yml").read_text(
        encoding="utf-8"
    )
    assert "bash scripts/check_bash_environment.sh" in bash_text
    assert "bash scripts/check_setup.sh --python-only" in bash_text
    assert "bash scripts/smoke_test_api.sh http://127.0.0.1:5000" in bash_text
    assert "bash-script-verification-logs" in bash_text

    security = load_workflow("security-scan.yml")
    security_text = Path(".github/workflows/security-scan.yml").read_text(encoding="utf-8")
    assert security["jobs"]["security-scan"]
    assert "pip_audit" in security_text
    assert "docker build -t mlops-flask-api:security" in security_text
    assert "trivy image" in security_text
    assert "security-reports" in security_text


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
    workflow_required_paths = [
        path
        for path in REQUIRED_COMMAND_PATHS
        if path != "scripts/final_readiness_check.py"
    ]
    for required_path in workflow_required_paths:
        assert Path(required_path).exists()
        assert required_path in workflow_text
    assert Path("scripts/final_readiness_check.py").exists()

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
    assert "COPY --chown=appuser:app data/raw/ data/raw/" in dockerfile
    assert "USER appuser" in dockerfile
    assert "useradd --system" in dockerfile
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
    assert "api_aware" in monitor_script
    assert "offline_simulated" in monitor_script
    assert "retraining_required" in monitor_script
    assert "data_quality_report.json" in monitor_script
    assert "retraining_required" in drift_script
    assert "drift_score" in drift_script
    assert "model_registry_summary" in registry


def test_quality_gate_includes_balanced_accuracy_contract() -> None:
    evaluate = Path("src/evaluate.py").read_text(encoding="utf-8")
    assert "MIN_BALANCED_ACCURACY" in evaluate
    assert "balanced_accuracy_above_minimum" in evaluate
    assert "candidate_balanced_accuracy" in evaluate
