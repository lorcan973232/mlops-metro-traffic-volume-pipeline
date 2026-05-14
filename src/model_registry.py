from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data import DATA_DOI, DATA_SOURCE_PAGE, write_json
from src.evaluate import METRICS_PATH, QUALITY_GATE_REPORT_PATH
from src.train import MODEL_PATH

MODEL_REGISTRY_PATH = Path("reports/metrics/model_registry.json")
MODEL_METADATA_PATH = Path("reports/metrics/model_metadata.json")


def register_model(
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    registry_path: Path = MODEL_REGISTRY_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> dict[str, Any]:
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
    record = {
        "model_version": metrics["model_version"],
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "dataset_source": DATA_SOURCE_PAGE,
        "dataset_doi": DATA_DOI,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "quality_gate_passed": quality_gate_report["passed"],
        "promotion_decision": (
            "accepted" if quality_gate_report["passed"] else "rejected"
        ),
    }
    metadata = {
        "model_version": metrics["model_version"],
        "dataset_name": metrics["dataset"].get("name"),
        "dataset_source": metrics["dataset"].get("source", DATA_SOURCE_PAGE),
        "dataset_hash": metrics["dataset"].get("raw_sha256"),
        "task_type": metrics.get("task_type", "multiclass_classification"),
        "feature_schema": metrics["feature_schema"],
        "class_labels": metrics.get("class_labels", []),
        "class_distribution": metrics.get("class_distribution", {}),
        "target_definition": metrics["target_definition"],
        "hyperparameters": metrics.get("hyperparameters", {}),
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
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "cohen_kappa": metrics.get("cohen_kappa"),
            "matthews_corrcoef": metrics.get("matthews_corrcoef"),
        },
        "per_class_metrics": metrics.get("per_class", {}),
        "probability_metrics": metrics.get("probability_metrics", {}),
        "confusion_matrix_path": metrics.get("confusion_matrix_path"),
        "confusion_matrix_normalized_path": metrics.get("confusion_matrix_normalized_path"),
        "classification_report_path": metrics.get("classification_report_json_path"),
        "model_path": str(model_path),
        "training_command": metrics["training_command"],
        "evaluation_command": metrics["evaluation_command"],
        "quality_gate": quality_gate_report,
    }
    write_json(registry_path, record)
    write_json(metadata_path, metadata)
    return record


def load_model_metadata(metadata_path: Path = MODEL_METADATA_PATH) -> dict[str, Any]:
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}. Run `python -m src.model_registry`."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def model_registry_summary(
    registry_path: Path = MODEL_REGISTRY_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> dict[str, Any]:
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
            "model_path": metadata["model_path"],
            "metrics_path": registry_record["metrics_path"],
            "training_timestamp": metadata["training_timestamp"],
            "metric_summary": metadata["metric_summary"],
            "hyperparameters": metadata.get("hyperparameters", {}),
            "per_class_metrics": metadata.get("per_class_metrics", {}),
            "quality_gate_passed": metadata["quality_gate"]["passed"],
        },
    }


def main() -> None:
    register_model()
    print(json.dumps(model_registry_summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
