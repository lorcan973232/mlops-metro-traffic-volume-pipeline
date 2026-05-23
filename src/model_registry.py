"""Record accepted model metadata for the repository-local registry.

The project does not use an external registry. This module writes the model,
metrics, quality gate, dataset, and feature-schema record under `reports/` so
Continuous Training, monitoring, README tables, and the demo all point to the
same accepted model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data import DATA_DOI, DATA_SOURCE_PAGE, file_sha256, write_json
from src.evaluate import METRICS_PATH, QUALITY_GATE_REPORT_PATH
from src.train import MODEL_PATH, PROCESSED_DATA_PATH
from src.versioning import get_current_version, register_version

MODEL_REGISTRY_PATH = Path("reports/metrics/model_registry.json")
MODEL_METADATA_PATH = Path("reports/metrics/model_metadata.json")
MODEL_REGISTRY_HISTORY_PATH = Path("reports/model_registry/version_history.json")


# ==============================================================================
# Lightweight model management
# ==============================================================================
#
# This is repository-based model management rather than an external registry.
# After evaluation, this file connects the accepted model, metrics, quality gate,
# dataset source, and feature schema. The outputs are saved under `reports/` so
# local runs, workflows, and the browser demo all refer to the same model record.


def register_model(
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    registry_path: Path = MODEL_REGISTRY_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    history_path: Path = MODEL_REGISTRY_HISTORY_PATH,
) -> dict[str, Any]:
    """Register the evaluated model only after metrics and gate reports exist."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    quality_gate_report = (
        json.loads(QUALITY_GATE_REPORT_PATH.read_text(encoding="utf-8"))
        if QUALITY_GATE_REPORT_PATH.exists()
        else metrics["quality_gate"]
    )

    # Keep registry metadata aligned with the evaluated metrics for reproducibility.
    is_accepted = quality_gate_report["passed"]
    model_version = metrics.get("model_version", get_current_version())

    # The processed-data hash ties the model record back to the exact dataset used
    # for training. Without this, a later run could look similar but be based on a
    # different local CSV.
    data_hash = file_sha256(PROCESSED_DATA_PATH) if PROCESSED_DATA_PATH.exists() else "unknown"

    record = {
        "model_version": model_version,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "dataset_source": DATA_SOURCE_PAGE,
        "dataset_doi": DATA_DOI,
        "task_type": metrics.get("task_type", "classification"),
        "accuracy": metrics["accuracy"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "f1_macro": metrics["f1_macro"],
        "quality_gate_passed": quality_gate_report["passed"],
        "promotion_decision": "accepted" if quality_gate_report["passed"] else "rejected",
    }
    version_history = {
        "registry_type": "lightweight_repository_metadata",
        "external_registry": False,
        "external_registry_claim": "none",
        "model_management_scope": (
            "Accepted model and metrics are stored as repository evidence and "
            "GitHub Actions artefacts. This is not an MLflow, GHCR, or cloud "
            "model-registry promotion."
        ),
        "records": [
            {
                "model_version": record["model_version"],
                "model_path": record["model_path"],
                "metrics_path": record["metrics_path"],
                "metadata_path": str(metadata_path),
                "dataset_source": record["dataset_source"],
                "dataset_doi": record["dataset_doi"],
                "quality_gate_passed": record["quality_gate_passed"],
                "promotion_decision": record["promotion_decision"],
                "training_timestamp": metrics["training_timestamp"],
                "accuracy": record["accuracy"],
                "f1_macro": record["f1_macro"],
                "f1_weighted": record["f1_weighted"],
            }
        ],
    }
    metadata = {
        "model_version": model_version,
        "dataset_name": metrics["dataset"].get("name"),
        "dataset_source": metrics["dataset"].get("source", DATA_SOURCE_PAGE),
        "dataset_hash": metrics["dataset"].get("raw_sha256"),
        "task_type": metrics.get("task_type", "classification"),
        "target_definition": metrics["target_definition"],
        "target_labels": metrics.get("target_labels", {}),
        "feature_schema": metrics["feature_schema"],
        "hyperparameters": metrics.get("hyperparameters", {}),
        "selected_model": metrics.get("selected_model", {}),
        "selected_hyperparameters": metrics.get("selected_hyperparameters", {}),
        "preprocessing_steps": metrics.get("hyperparameters", {}).get("preprocessing", {}),
        "train_test_split": metrics.get("hyperparameters", {}).get("train_test_split", {}),
        "random_state": metrics.get("hyperparameters", {})
        .get("classifier", {})
        .get("random_state"),
        "cross_validation_method": metrics.get("cross_validation", {}).get("method"),
        "scoring_metrics": metrics.get("cross_validation", {}).get("scoring", []),
        "training_timestamp": metrics["training_timestamp"],
        "metric_summary": {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_weighted": metrics["f1_weighted"],
            "roc_auc": metrics["roc_auc"],
        },
        "confusion_matrix": metrics.get("confusion_matrix", []),
        "model_path": str(model_path),
        "training_command": metrics["training_command"],
        "evaluation_command": metrics["evaluation_command"],
        "quality_gate": quality_gate_report,
    }

    # Only accepted models become active rollback candidates. Rejected candidates
    # still have metrics, but they are not promoted by Continuous Training.
    if is_accepted:
        register_version(
            version=model_version,
            model_path=model_path,
            metadata_path=metadata_path,
            metrics=metadata["metric_summary"],
            data_hash=data_hash,
            training_timestamp=metrics["training_timestamp"],
            quality_gate_passed=True,
        )

    write_json(registry_path, record)
    write_json(metadata_path, metadata)
    write_json(history_path, version_history)
    return record


def load_model_metadata(metadata_path: Path = MODEL_METADATA_PATH) -> dict[str, Any]:
    """Load metadata used by monitoring, readiness checks, and the demo."""
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}. Run `python -m src.model_registry`."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def model_registry_summary(
    registry_path: Path = MODEL_REGISTRY_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> dict[str, Any]:
    """Build a compact registry view for CLI output and quick inspection."""
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Model registry not found: {registry_path}. Run `python -m src.model_registry`."
        )
    registry_record = json.loads(registry_path.read_text(encoding="utf-8"))
    metadata = load_model_metadata(metadata_path)
    return {
        "registry_record": registry_record,
        "model_metadata": {
            "model_version": metadata["model_version"],
            "dataset_name": metadata["dataset_name"],
            "dataset_source": metadata["dataset_source"],
            "feature_schema": metadata["feature_schema"],
            "target_definition": metadata["target_definition"],
            "target_labels": metadata.get("target_labels", {}),
            "model_path": metadata["model_path"],
            "metrics_path": registry_record["metrics_path"],
            "training_timestamp": metadata["training_timestamp"],
            "metric_summary": metadata["metric_summary"],
            "hyperparameters": metadata.get("hyperparameters", {}),
            "selected_model": metadata.get("selected_model", {}),
            "selected_hyperparameters": metadata.get("selected_hyperparameters", {}),
            "quality_gate_passed": metadata["quality_gate"]["passed"],
        },
        "version_history_path": str(MODEL_REGISTRY_HISTORY_PATH),
    }


def main() -> None:
    """CLI entry point used locally and by GitHub Actions after evaluation."""
    register_model()
    print(json.dumps(model_registry_summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
