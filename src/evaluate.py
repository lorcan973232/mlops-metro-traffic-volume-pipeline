from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.data import CLASS_COLUMN, CLASS_LABELS, FEATURE_COLUMNS, write_json
from src.preprocess import PROCESSED_DATA_PATH
from src.train import MODEL_PATH, RANDOM_STATE, TEST_SIZE, load_processed_data, train_model

METRICS_PATH = Path("reports/metrics/metrics.json")
BASELINE_METRICS_PATH = Path("reports/metrics/baseline_metrics.json")
LATEST_METRICS_PATH = Path("reports/metrics/latest_metrics.json")
QUALITY_GATE_PATH = Path("reports/metrics/quality_gate.json")
QUALITY_GATE_REPORT_PATH = Path("reports/metrics/quality_gate_report.json")
MIN_ACCURACY = 0.50
MIN_MACRO_F1 = 0.35
MAX_BASELINE_REGRESSION = 0.02
EVALUATION_COMMAND = "python -m src.evaluate"


def _metric_summary(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
    }


def load_or_create_baseline(latest_metrics: dict[str, Any]) -> dict[str, Any]:
    latest_summary = _metric_summary(latest_metrics)
    if BASELINE_METRICS_PATH.exists():
        return json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))

    baseline = {
        "status": "baseline_established",
        "source": "first_verified_local_evaluation",
        "model_version": latest_metrics["model_version"],
        "dataset": latest_metrics["dataset"],
        "metric_summary": latest_summary,
        "quality_thresholds": {
            "min_accuracy": MIN_ACCURACY,
            "min_macro_f1": MIN_MACRO_F1,
            "max_baseline_regression": MAX_BASELINE_REGRESSION,
        },
        "created_by": EVALUATION_COMMAND,
    }
    write_json(BASELINE_METRICS_PATH, baseline)
    return baseline


def evaluate_model(
    model_path: Path = MODEL_PATH,
    processed_path: Path = PROCESSED_DATA_PATH,
    min_accuracy: float = MIN_ACCURACY,
    min_macro_f1: float = MIN_MACRO_F1,
) -> dict[str, Any]:
    if not model_path.exists():
        train_model(processed_path=processed_path, model_path=model_path)

    data = load_processed_data(processed_path)
    x = data[FEATURE_COLUMNS]
    y = data[CLASS_COLUMN]
    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    bundle = joblib.load(model_path)
    predictions = bundle["model"].predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    macro_f1 = float(f1_score(y_test, predictions, average="macro", zero_division=0))
    metrics = {
        "status": "evaluated",
        "model_version": bundle.get("model_version", "unknown"),
        "model_path": str(model_path),
        "dataset": bundle.get("dataset", {}),
        "target_definition": bundle.get("target_definition", {}),
        "feature_schema": bundle.get("feature_columns", FEATURE_COLUMNS),
        "training_timestamp": bundle.get("training_timestamp"),
        "training_command": bundle.get("training_command", "python -m src.train"),
        "evaluation_command": EVALUATION_COMMAND,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=list(CLASS_LABELS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=list(CLASS_LABELS),
        ).tolist(),
    }
    latest_metrics = {
        "status": "latest_evaluation",
        "model_version": metrics["model_version"],
        "model_path": metrics["model_path"],
        "dataset": metrics["dataset"],
        "feature_schema": metrics["feature_schema"],
        "metric_summary": _metric_summary(metrics),
        "training_timestamp": metrics["training_timestamp"],
        "training_command": metrics["training_command"],
        "evaluation_command": EVALUATION_COMMAND,
    }
    write_json(LATEST_METRICS_PATH, latest_metrics)
    baseline = load_or_create_baseline(metrics)
    baseline_summary = baseline["metric_summary"]
    threshold_checks = {
        "accuracy_above_minimum": accuracy >= min_accuracy,
        "macro_f1_above_minimum": macro_f1 >= min_macro_f1,
        "accuracy_not_regressed_from_baseline": (
            accuracy >= float(baseline_summary["accuracy"]) - MAX_BASELINE_REGRESSION
        ),
        "macro_f1_not_regressed_from_baseline": (
            macro_f1 >= float(baseline_summary["macro_f1"]) - MAX_BASELINE_REGRESSION
        ),
    }
    gate_passed = all(threshold_checks.values())
    quality_gate = {
        "status": "passed" if gate_passed else "failed",
        "passed": gate_passed,
        "thresholds": {
            "min_accuracy": min_accuracy,
            "min_macro_f1": min_macro_f1,
            "max_baseline_regression": MAX_BASELINE_REGRESSION,
        },
        "latest_metrics": latest_metrics["metric_summary"],
        "baseline_metrics": baseline_summary,
        "checks": threshold_checks,
        "decision": "accept_candidate_model" if gate_passed else "reject_candidate_model",
    }
    metrics["quality_gate"] = quality_gate
    write_json(METRICS_PATH, metrics)
    write_json(QUALITY_GATE_PATH, quality_gate)
    write_json(QUALITY_GATE_REPORT_PATH, quality_gate)
    if not gate_passed:
        raise RuntimeError(
            f"Quality gate failed: accuracy={accuracy:.3f}, macro_f1={macro_f1:.3f}."
        )
    return metrics


def main() -> None:
    metrics = evaluate_model()
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
