"""Tests that committed evidence files match the README and workflow contract.

These tests are not checking model quality directly. They protect the project
against a marking-risk problem: reports, workflow names, and demo paths can drift
away from the README even when the code still runs.
"""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_METRIC_FILES = [
    "metrics.json",
    "metrics.csv",
    "latest_metrics.json",
    "baseline_metrics.json",
    "quality_gate_report.json",
    "model_metadata.json",
    "final_model_metadata.json",
    "model_comparison.json",
    "hyperparameter_search_results.json",
    "hyperparameter_results.csv",
    "best_params.json",
    "classification_report.json",
    "classification_report.txt",
    "confusion_matrix.json",
    "confusion_matrix_normalized.json",
    "confusion_matrix.png",
    "cross_validation_results.json",
    "cross_validation_results.csv",
    "error_analysis.json",
    "feature_importance.json",
    "fairness_analysis.json",
    "ensemble_comparison.json",
]

REQUIRED_WORKFLOW_FILES = [
    "ci.yml",
    "data-preprocessing.yml",
    "train-and-evaluate.yml",
    "docker-build.yml",
    "deploy.yml",
    "continuous-training.yml",
    "monitoring.yml",
    "model-analysis.yml",
    "repository-visibility-check.yml",
]


def test_full_metrics_and_model_management_package_is_present() -> None:
    """Check the metric and metadata files a marker opens during review exist."""
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
    assert metadata["model_path"].replace("\\", "/") == "models/traffic_volume_classifier.joblib"
    assert metadata["hyperparameters"]["algorithm"] == "HistGradientBoostingClassifier"
    assert metadata["feature_schema"] == latest["feature_schema"]
    assert isinstance(gate["checks"]["balanced_accuracy_above_minimum"], bool)
    assert gate["thresholds"]["min_accuracy"] == 0.975
    assert gate["passed"] is True
    assert gate["decision"] == "accept_candidate_model"
    assert gate["decision"] in {"accept_candidate_model", "reject_candidate_model"}


def test_feature_importance_and_fairness_analysis_are_present() -> None:
    """Check feature-importance and class-balance reports have useful structure."""
    metrics_dir = Path("reports/metrics")
    feature_importance_path = metrics_dir / "feature_importance.json"
    fairness_analysis_path = metrics_dir / "fairness_analysis.json"

    assert feature_importance_path.is_file(), "feature_importance.json missing"
    assert fairness_analysis_path.is_file(), "fairness_analysis.json missing"

    feature_imp = json.loads(feature_importance_path.read_text(encoding="utf-8"))
    fairness = json.loads(fairness_analysis_path.read_text(encoding="utf-8"))

    # Feature importance must be model-derived evidence, not an unsupported claim.
    assert feature_imp["status"] == "computed"
    assert "algorithm" in feature_imp
    assert "features" in feature_imp
    assert "top_3_features" in feature_imp
    assert len(feature_imp["top_3_features"]) == 3
    assert all(isinstance(f[1], float) for f in feature_imp["top_3_features"])

    # This legacy class-balance file should still expose both prediction classes.
    assert fairness["status"] == "fairness_analyzed"
    assert "per_class_metrics" in fairness
    assert "disparities" in fairness
    assert "is_balanced" in fairness
    assert isinstance(fairness["is_balanced"], bool)
    assert "normal traffic" in fairness["per_class_metrics"]
    assert "high traffic" in fairness["per_class_metrics"]
    for class_name in ["normal traffic", "high traffic"]:
        class_metrics = fairness["per_class_metrics"][class_name]
        assert "precision" in class_metrics
        assert "recall" in class_metrics
        assert "f1_score" in class_metrics
        assert "support" in class_metrics


def test_readme_exposes_marker_facing_artefact_evidence() -> None:
    """Check the README points examiners to lifecycle evidence and demo paths.

    The README was simplified into human coursework wording, so this test checks
    the new section names and saved evidence paths rather than the older formal
    headings or date wording.
    """
    readme = Path("README.md").read_text(encoding="utf-8")

    required_sections = [
        "Public GitHub repository:",
        "reports/submission/public_repository_evidence.json",
        "reports/submission/branching_evidence.md",
        "## GitHub Actions workflows",
        "## Extra evidence, if included",
        "## Branching strategy",
        "## Demo steps",
        "## Traceability table",
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
    """Check demo scripts cover local Python, Windows, Docker, and Kind routes."""
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
        "scripts/explain_model.py",
        "scripts/fairness_audit.py",
        "scripts/cost_benefit_analysis.py",
    ]:
        assert stage in powershell_runner


def test_required_workflows_exist_and_upload_marker_evidence() -> None:
    """Check expected workflow artefact names remain present in committed YAML."""
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
        "tier3-model-analysis-reports",
        "repository-visibility-evidence",
    ]:
        assert artefact_name in workflow_text

    assert "kind create cluster" in workflow_text
    assert "kubectl rollout status" in workflow_text
    assert "Continuous Training rejected candidate model." in workflow_text
    assert "retraining_required" in workflow_text


def test_public_repository_submission_evidence_is_present() -> None:
    """Check public repository evidence is present and states the required date."""
    evidence_path = Path("reports/submission/public_repository_evidence.json")
    branching_path = Path("reports/submission/branching_evidence.md")
    assert evidence_path.is_file()
    assert branching_path.is_file()

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["repository_url"] == (
        "https://github.com/lorcan973232/mlops-wine-quality-pipeline"
    )
    assert evidence["visibility"] == "public"
    assert evidence["private"] is False
    assert "21 June 2026" in evidence["requirement_note"]
