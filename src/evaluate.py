from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import label_binarize

from src.data import CLASS_COLUMN, CLASS_LABELS, FEATURE_COLUMNS, write_json
from src.preprocess import PROCESSED_DATA_PATH
from src.train import MODEL_PATH, RANDOM_STATE, TEST_SIZE, load_processed_data, train_model

METRICS_PATH = Path("reports/metrics/metrics.json")
BASELINE_METRICS_PATH = Path("reports/metrics/baseline_metrics.json")
LATEST_METRICS_PATH = Path("reports/metrics/latest_metrics.json")
QUALITY_GATE_PATH = Path("reports/metrics/quality_gate.json")
QUALITY_GATE_REPORT_PATH = Path("reports/metrics/quality_gate_report.json")
CLASSIFICATION_REPORT_JSON_PATH = Path("reports/metrics/classification_report.json")
CLASSIFICATION_REPORT_TEXT_PATH = Path("reports/metrics/classification_report.txt")
CONFUSION_MATRIX_PATH = Path("reports/metrics/confusion_matrix.json")
CONFUSION_MATRIX_NORMALIZED_PATH = Path("reports/metrics/confusion_matrix_normalized.json")
CROSS_VALIDATION_RESULTS_PATH = Path("reports/metrics/cross_validation_results.json")

MIN_ACCURACY = 0.94
MIN_MACRO_F1 = 0.94
MIN_BALANCED_ACCURACY = 0.94
MIN_WEIGHTED_F1 = 0.94
MAX_BASELINE_REGRESSION = 0.02
CV_SPLITS = 5
EVALUATION_COMMAND = "python -m src.evaluate"


def _safe_metric(value: float | str) -> float | str:
    if isinstance(value, str):
        return value
    if np.isnan(value):
        return "NOT_APPLICABLE"
    return float(value)


def _metric_summary(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_precision": float(metrics["weighted_precision"]),
        "weighted_recall": float(metrics["weighted_recall"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "micro_f1": float(metrics["micro_f1"]),
        "cohen_kappa": float(metrics["cohen_kappa"]),
        "matthews_corrcoef": float(metrics["matthews_corrcoef"]),
    }


def _ordered_probabilities(model: Any, probabilities: np.ndarray) -> np.ndarray | None:
    if not hasattr(model, "classes_"):
        return None
    model_classes = list(model.classes_)
    if not set(CLASS_LABELS).issubset(model_classes):
        return None
    ordered_indices = [model_classes.index(label) for label in CLASS_LABELS]
    return probabilities[:, ordered_indices]


def _probability_metrics(y_true: Any, probabilities: np.ndarray | None) -> dict[str, Any]:
    if probabilities is None:
        return {
            "roc_auc_ovr_macro": "NOT_APPLICABLE",
            "roc_auc_ovr_weighted": "NOT_APPLICABLE",
            "average_precision_macro": "NOT_APPLICABLE",
            "average_precision_weighted": "NOT_APPLICABLE",
            "log_loss": "NOT_APPLICABLE",
            "top_2_accuracy": "NOT_APPLICABLE",
            "not_applicable_reason": "Selected estimator does not expose class probabilities.",
        }
    label_to_index = {label: index for index, label in enumerate(CLASS_LABELS)}
    y_encoded = np.array([label_to_index[label] for label in y_true])
    encoded_labels = list(range(len(CLASS_LABELS)))
    y_binary = label_binarize(y_encoded, classes=encoded_labels)
    if len(CLASS_LABELS) == 2:
        y_binary = np.column_stack([1 - y_binary.ravel(), y_binary.ravel()])
        return {
            "roc_auc_ovr_macro": _safe_metric(
                roc_auc_score(y_binary, probabilities, average="macro")
            ),
            "roc_auc_ovr_weighted": _safe_metric(
                roc_auc_score(y_binary, probabilities, average="weighted")
            ),
            "average_precision_macro": _safe_metric(
                average_precision_score(y_binary, probabilities, average="macro")
            ),
            "average_precision_weighted": _safe_metric(
                average_precision_score(y_binary, probabilities, average="weighted")
            ),
            "log_loss": _safe_metric(log_loss(y_encoded, probabilities, labels=encoded_labels)),
            "top_2_accuracy": "NOT_APPLICABLE",
            "not_applicable_reason": "Top-2 accuracy is not meaningful for a binary classifier.",
        }
    return {
        "roc_auc_ovr_macro": _safe_metric(
            roc_auc_score(y_encoded, probabilities, labels=encoded_labels, multi_class="ovr")
        ),
        "roc_auc_ovr_weighted": _safe_metric(
            roc_auc_score(
                y_encoded,
                probabilities,
                labels=encoded_labels,
                multi_class="ovr",
                average="weighted",
            )
        ),
        "average_precision_macro": _safe_metric(
            average_precision_score(y_binary, probabilities, average="macro")
        ),
        "average_precision_weighted": _safe_metric(
            average_precision_score(y_binary, probabilities, average="weighted")
        ),
        "log_loss": _safe_metric(log_loss(y_encoded, probabilities, labels=encoded_labels)),
        "top_2_accuracy": _safe_metric(
            top_k_accuracy_score(y_encoded, probabilities, k=2, labels=encoded_labels)
        ),
    }


def _basic_metrics(y_true: Any, predictions: Any) -> dict[str, Any]:
    report = classification_report(
        y_true,
        predictions,
        labels=list(CLASS_LABELS),
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in CLASS_LABELS
    }
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_precision": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_recall": float(
            recall_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true, predictions, average="micro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, predictions, labels=list(CLASS_LABELS))),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, predictions)),
        "per_class": per_class,
        "classification_report": report,
        "classification_report_text": classification_report(
            y_true,
            predictions,
            labels=list(CLASS_LABELS),
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=list(CLASS_LABELS),
        ).tolist(),
        "confusion_matrix_normalized": confusion_matrix(
            y_true,
            predictions,
            labels=list(CLASS_LABELS),
            normalize="true",
        ).tolist(),
    }


def _evaluate_baseline(x_train: Any, y_train: Any, x_test: Any, y_test: Any) -> dict[str, Any]:
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    predictions = baseline.predict(x_test)
    basic = _basic_metrics(y_test, predictions)
    return {
        "status": "baseline_evaluated",
        "baseline_model": "DummyClassifier(strategy='most_frequent')",
        "model_version": "dummy-most-frequent-baseline",
        "metric_summary": {
            "accuracy": basic["accuracy"],
            "balanced_accuracy": basic["balanced_accuracy"],
            "macro_f1": basic["macro_f1"],
            "weighted_f1": basic["weighted_f1"],
        },
        "quality_thresholds": {
            "min_accuracy": MIN_ACCURACY,
            "min_macro_f1": MIN_MACRO_F1,
            "min_balanced_accuracy": MIN_BALANCED_ACCURACY,
            "min_weighted_f1": MIN_WEIGHTED_F1,
            "max_baseline_regression": MAX_BASELINE_REGRESSION,
        },
        "created_by": EVALUATION_COMMAND,
    }


def _cross_validation_report(estimator: Any, x_train: Any, y_train: Any) -> dict[str, Any]:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "macro_f1": "f1_macro",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
    }
    results = cross_validate(estimator, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1)

    def fold_summary(key: str) -> dict[str, Any]:
        values = results[f"test_{key}"]
        return {
            "per_fold": [float(value) for value in values],
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    return {
        "status": "completed",
        "method": "StratifiedKFold",
        "folds": CV_SPLITS,
        "shuffle": True,
        "random_state": RANDOM_STATE,
        "scoring": list(scoring),
        "macro_f1": fold_summary("macro_f1"),
        "accuracy": fold_summary("accuracy"),
        "balanced_accuracy": fold_summary("balanced_accuracy"),
    }


def _quality_gate(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    baseline_summary = baseline["metric_summary"]
    checks = {
        "accuracy_above_minimum": metrics["accuracy"] >= MIN_ACCURACY,
        "macro_f1_above_minimum": metrics["macro_f1"] >= MIN_MACRO_F1,
        "balanced_accuracy_above_minimum": metrics["balanced_accuracy"] >= MIN_BALANCED_ACCURACY,
        "weighted_f1_above_minimum": metrics["weighted_f1"] >= MIN_WEIGHTED_F1,
        "accuracy_above_baseline": metrics["accuracy"]
        >= float(baseline_summary["accuracy"]) - MAX_BASELINE_REGRESSION,
        "macro_f1_above_baseline": metrics["macro_f1"]
        >= float(baseline_summary["macro_f1"]) - MAX_BASELINE_REGRESSION,
        "balanced_accuracy_above_baseline": metrics["balanced_accuracy"]
        >= float(baseline_summary["balanced_accuracy"]) - MAX_BASELINE_REGRESSION,
    }
    passed = all(checks.values())
    failed_checks = [check for check, result in checks.items() if not result]
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "decision": "accept_candidate_model" if passed else "reject_candidate_model",
        "thresholds": baseline["quality_thresholds"],
        "candidate_accuracy": metrics["accuracy"],
        "candidate_macro_f1": metrics["macro_f1"],
        "candidate_balanced_accuracy": metrics["balanced_accuracy"],
        "candidate_weighted_f1": metrics["weighted_f1"],
        "baseline_accuracy": baseline_summary["accuracy"],
        "baseline_macro_f1": baseline_summary["macro_f1"],
        "baseline_balanced_accuracy": baseline_summary["balanced_accuracy"],
        "checks": checks,
        "reasons": (
            ["All model performance quality gates passed."]
            if passed
            else [f"Failed check: {check}" for check in failed_checks]
        ),
    }


def evaluate_model(
    model_path: Path = MODEL_PATH,
    processed_path: Path = PROCESSED_DATA_PATH,
) -> dict[str, Any]:
    if not model_path.exists():
        train_model(processed_path=processed_path, model_path=model_path)

    data = load_processed_data(processed_path)
    x = data[FEATURE_COLUMNS]
    y = data[CLASS_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    bundle = joblib.load(model_path)
    model = bundle["model"]
    predictions = model.predict(x_test)
    probabilities = (
        _ordered_probabilities(model, model.predict_proba(x_test))
        if hasattr(model, "predict_proba")
        else None
    )
    basic = _basic_metrics(y_test, predictions)
    probability_metrics = _probability_metrics(y_test, probabilities)
    baseline = _evaluate_baseline(x_train, y_train, x_test, y_test)
    cv_report = _cross_validation_report(model, x_train, y_train)

    metrics = {
        "status": "evaluated",
        "model_version": bundle.get("model_version", "unknown"),
        "model_path": str(model_path),
        "task_type": bundle.get("task_type", "multiclass_classification"),
        "dataset": bundle.get("dataset", {}),
        "class_labels": list(CLASS_LABELS),
        "target_definition": bundle.get("target_definition", {}),
        "feature_schema": bundle.get("feature_columns", FEATURE_COLUMNS),
        "class_distribution": bundle.get("class_distribution", {}),
        "hyperparameters": bundle.get("hyperparameters", {}),
        "training_timestamp": bundle.get("training_timestamp"),
        "training_command": bundle.get("training_command", "python -m src.train"),
        "evaluation_command": EVALUATION_COMMAND,
        **{
            key: value
            for key, value in basic.items()
            if key
            not in {
                "classification_report_text",
            }
        },
        "classification_report_text_path": str(CLASSIFICATION_REPORT_TEXT_PATH),
        "classification_report_json_path": str(CLASSIFICATION_REPORT_JSON_PATH),
        "confusion_matrix_path": str(CONFUSION_MATRIX_PATH),
        "confusion_matrix_normalized_path": str(CONFUSION_MATRIX_NORMALIZED_PATH),
        "probability_metrics": probability_metrics,
        "cross_validation": cv_report,
        "baseline": baseline,
    }
    metrics["quality_gate"] = _quality_gate(metrics, baseline)

    latest_metrics = {
        "status": "latest_evaluation",
        "model_version": metrics["model_version"],
        "model_path": metrics["model_path"],
        "dataset": metrics["dataset"],
        "task_type": metrics["task_type"],
        "feature_schema": metrics["feature_schema"],
        "class_labels": metrics["class_labels"],
        "hyperparameters": metrics["hyperparameters"],
        "metric_summary": _metric_summary(metrics),
        "probability_metrics": probability_metrics,
        "quality_gate": metrics["quality_gate"],
        "training_timestamp": metrics["training_timestamp"],
        "training_command": metrics["training_command"],
        "evaluation_command": EVALUATION_COMMAND,
    }
    write_json(LATEST_METRICS_PATH, latest_metrics)
    write_json(BASELINE_METRICS_PATH, baseline)
    write_json(CLASSIFICATION_REPORT_JSON_PATH, basic["classification_report"])
    CLASSIFICATION_REPORT_TEXT_PATH.write_text(
        basic["classification_report_text"] + "\n",
        encoding="utf-8",
    )
    write_json(
        CONFUSION_MATRIX_PATH,
        {"class_labels": list(CLASS_LABELS), "matrix": basic["confusion_matrix"]},
    )
    write_json(
        CONFUSION_MATRIX_NORMALIZED_PATH,
        {
            "class_labels": list(CLASS_LABELS),
            "normalization": "true_label_row_normalized",
            "matrix": basic["confusion_matrix_normalized"],
        },
    )
    write_json(CROSS_VALIDATION_RESULTS_PATH, cv_report)
    write_json(METRICS_PATH, metrics)
    write_json(QUALITY_GATE_PATH, metrics["quality_gate"])
    write_json(QUALITY_GATE_REPORT_PATH, metrics["quality_gate"])
    if not metrics["quality_gate"]["passed"]:
        raise RuntimeError(f"Quality gate failed: {metrics['quality_gate']['reasons']}")
    return metrics


def main() -> None:
    metrics = evaluate_model()
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
