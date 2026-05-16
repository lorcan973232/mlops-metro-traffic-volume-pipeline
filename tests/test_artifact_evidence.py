from __future__ import annotations

import json
from pathlib import Path

REQUIRED_METRIC_FILES = [
    "latest_metrics.json",
    "baseline_metrics.json",
    "quality_gate_report.json",
    "model_metadata.json",
    "model_comparison.json",
    "hyperparameter_search_results.json",
    "classification_report.json",
    "classification_report.txt",
    "confusion_matrix.json",
    "confusion_matrix_normalized.json",
    "cross_validation_results.json",
]

REQUIRED_WORKFLOW_FILES = [
    "ci.yml",
    "data-preprocessing.yml",
    "train-and-evaluate.yml",
    "docker-build.yml",
    "deploy.yml",
    "continuous-training.yml",
    "monitoring.yml",
]


def test_full_metrics_and_model_management_package_is_present() -> None:
    metrics_dir = Path("reports/metrics")
    missing_files = [name for name in REQUIRED_METRIC_FILES if not (metrics_dir / name).is_file()]
    assert missing_files == []

    latest = json.loads((metrics_dir / "latest_metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads((metrics_dir / "model_metadata.json").read_text(encoding="utf-8"))
    gate = json.loads((metrics_dir / "quality_gate_report.json").read_text(encoding="utf-8"))

    for metric_name in [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]:
        assert metric_name in latest["metric_summary"]

    assert metadata["model_version"] == latest["model_version"]
    assert metadata["model_path"].replace("\\", "/") == "models/wine_quality_classifier.joblib"
    assert metadata["hyperparameters"]["algorithm"] == "ExtraTreesClassifier"
    assert metadata["feature_schema"] == latest["feature_schema"]
    assert gate["checks"]["balanced_accuracy_above_minimum"] is True
    assert gate["decision"] in {"accept_candidate_model", "reject_candidate_model"}


def test_readme_exposes_marker_facing_artefact_evidence() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    required_sections = [
        "Public GitHub repository:",
        "remain public until 21 June 2026",
        "## MLOps Workflow Detail: CI/CD/CT/CM",
        "## Branching Strategy",
        "## Live Demo Checklist",
        "## Traceability Matrix",
    ]
    for section in required_sections:
        assert section in readme

    for workflow_name in [
        "CI",
        "Data Preprocessing",
        "Train and Evaluate",
        "Docker Build",
        "Deploy Kind",
        "Continuous Training",
        "Monitoring",
    ]:
        assert workflow_name in readme


def test_live_demo_scripts_cover_python_windows_docker_and_kind_paths() -> None:
    required_scripts = [
        "scripts/run_pipeline.sh",
        "scripts/run_pipeline.ps1",
        "scripts/smoke_test_api.sh",
        "scripts/smoke_test_api.ps1",
        "scripts/create_kind_cluster.sh",
        "scripts/create_kind_cluster.ps1",
        "scripts/deploy_kind.sh",
        "scripts/deploy_kind.ps1",
    ]
    for script_path in required_scripts:
        assert Path(script_path).is_file()

    bash_runner = Path("scripts/run_pipeline.sh").read_text(encoding="utf-8")
    powershell_runner = Path("scripts/run_pipeline.ps1").read_text(encoding="utf-8")
    for command in [
        "python -m compileall app src tests",
        "python -m src.predict",
        "pytest -q",
        "ruff check src tests",
        "from app.main import app",
    ]:
        assert command in bash_runner

    for stage in [
        "src.predict",
        "pytest",
        "ruff",
        "scripts/monitor.py",
        "scripts/check_drift.py",
    ]:
        assert stage in powershell_runner


def test_required_workflows_exist_and_upload_marker_evidence() -> None:
    workflow_dir = Path(".github/workflows")
    missing_workflows = [
        name for name in REQUIRED_WORKFLOW_FILES if not (workflow_dir / name).is_file()
    ]
    assert missing_workflows == []

    workflow_text = "\n".join(
        (workflow_dir / name).read_text(encoding="utf-8") for name in REQUIRED_WORKFLOW_FILES
    )
    for artefact_name in [
        "ci-artifacts",
        "preprocessing-artifacts",
        "train-evaluate-artifacts",
        "docker-smoke-logs",
        "kind-deployment-logs",
        "continuous-training-artifacts",
        "monitoring-artifacts",
    ]:
        assert artefact_name in workflow_text

    assert "kind create cluster" in workflow_text
    assert "kubectl rollout status" in workflow_text
    assert "Continuous Training rejected candidate model." in workflow_text
    assert "retraining_required" in workflow_text
