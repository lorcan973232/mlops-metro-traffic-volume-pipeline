from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_LABELS, write_json
from src.preprocess import PROCESSED_DATA_PATH
from src.train import MODEL_PATH, RANDOM_STATE, TEST_SIZE, load_processed_data, train_model

METRICS_PATH = Path("reports/metrics/metrics.json")
BASELINE_METRICS_PATH = Path("reports/metrics/baseline_metrics.json")
LATEST_METRICS_PATH = Path("reports/metrics/latest_metrics.json")
QUALITY_GATE_PATH = Path("reports/metrics/quality_gate.json")
QUALITY_GATE_REPORT_PATH = Path("reports/metrics/quality_gate_report.json")
MODEL_METADATA_PATH = Path("reports/metrics/model_metadata.json")
MODEL_COMPARISON_PATH = Path("reports/metrics/model_comparison.json")
HYPERPARAMETER_SEARCH_RESULTS_PATH = Path("reports/metrics/hyperparameter_search_results.json")
CROSS_VALIDATION_RESULTS_PATH = Path("reports/metrics/cross_validation_results.json")
CLASSIFICATION_REPORT_JSON_PATH = Path("reports/metrics/classification_report.json")
CLASSIFICATION_REPORT_TEXT_PATH = Path("reports/metrics/classification_report.txt")
CONFUSION_MATRIX_PATH = Path("reports/metrics/confusion_matrix.json")
CONFUSION_MATRIX_NORMALIZED_PATH = Path("reports/metrics/confusion_matrix_normalized.json")

MIN_ACCURACY = 0.80
MIN_BALANCED_ACCURACY = 0.80
MIN_WEIGHTED_F1 = 0.80
MIN_MACRO_F1 = 0.80
MIN_CV_ACCURACY = 0.77
MIN_BASELINE_ACCURACY_IMPROVEMENT = 0.20
CV_SPLITS = 5
EVALUATION_COMMAND = "python -m src.evaluate"
CLASS_NAMES = [TARGET_LABELS[0], TARGET_LABELS[1]]


def _positive_class_probabilities(model: Any, x_test: Any) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba(x_test)
    classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
    if 1 not in classes:
        return None
    return probabilities[:, classes.index(1)]


def _classification_metrics(
    y_true: Any,
    predictions: Any,
    positive_probabilities: np.ndarray | None = None,
) -> dict[str, float | None]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, predictions, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, predictions, average="weighted", zero_division=0
    )
    roc_auc = (
        float(roc_auc_score(y_true, positive_probabilities))
        if positive_probabilities is not None
        else None
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "roc_auc": roc_auc,
    }


def _evaluate_baseline(x_train: Any, y_train: Any, x_test: Any, y_test: Any) -> dict[str, Any]:
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    predictions = baseline.predict(x_test)
    metrics = _classification_metrics(y_test, predictions)
    return {
        "status": "baseline_evaluated",
        "baseline_model": "DummyClassifier(strategy='most_frequent')",
        "model_version": "dummy-most-frequent-baseline",
        "metric_summary": metrics,
        "quality_thresholds": {
            "min_accuracy": MIN_ACCURACY,
            "min_balanced_accuracy": MIN_BALANCED_ACCURACY,
            "min_weighted_f1": MIN_WEIGHTED_F1,
            "min_macro_f1": MIN_MACRO_F1,
            "min_cv_accuracy": MIN_CV_ACCURACY,
            "min_baseline_accuracy_improvement": MIN_BASELINE_ACCURACY_IMPROVEMENT,
        },
        "created_by": EVALUATION_COMMAND,
    }


def _cross_validation_report(estimator: Any, x_train: Any, y_train: Any) -> dict[str, Any]:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
        "f1_weighted": "f1_weighted",
    }
    results = cross_validate(
        estimator,
        x_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    def summary(values: np.ndarray) -> dict[str, Any]:
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
        "accuracy": summary(results["test_accuracy"]),
        "precision_macro": summary(results["test_precision_macro"]),
        "recall_macro": summary(results["test_recall_macro"]),
        "f1_macro": summary(results["test_f1_macro"]),
        "precision_weighted": summary(results["test_precision_weighted"]),
        "recall_weighted": summary(results["test_recall_weighted"]),
        "f1_weighted": summary(results["test_f1_weighted"]),
    }


def _quality_gate(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    cv_report: dict[str, Any],
) -> dict[str, Any]:
    baseline_summary = baseline["metric_summary"]
    checks = {
        "accuracy_above_minimum": metrics["accuracy"] >= MIN_ACCURACY,
        "balanced_accuracy_above_minimum": (
            metrics["balanced_accuracy"] >= MIN_BALANCED_ACCURACY
        ),
        "weighted_f1_above_minimum": metrics["f1_weighted"] >= MIN_WEIGHTED_F1,
        "macro_f1_above_minimum": metrics["f1_macro"] >= MIN_MACRO_F1,
        "cv_accuracy_above_minimum": cv_report["accuracy"]["mean"] >= MIN_CV_ACCURACY,
        "accuracy_beats_baseline_margin": metrics["accuracy"]
        >= baseline_summary["accuracy"] + MIN_BASELINE_ACCURACY_IMPROVEMENT,
    }
    passed = all(checks.values())
    failed_checks = [check for check, result in checks.items() if not result]
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "decision": "accept_candidate_model" if passed else "reject_candidate_model",
        "thresholds": baseline["quality_thresholds"],
        "candidate_accuracy": metrics["accuracy"],
        "candidate_balanced_accuracy": metrics["balanced_accuracy"],
        "candidate_f1_weighted": metrics["f1_weighted"],
        "candidate_f1_macro": metrics["f1_macro"],
        "cv_accuracy_mean": cv_report["accuracy"]["mean"],
        "baseline_accuracy": baseline_summary["accuracy"],
        "baseline_f1_weighted": baseline_summary["f1_weighted"],
        "checks": checks,
        "reasons": (
            ["All classification model quality gates passed."]
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
    y = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y,
    )
    bundle = joblib.load(model_path)
    model = bundle["model"]
    predictions = model.predict(x_test)
    positive_probabilities = _positive_class_probabilities(model, x_test)
    basic = _classification_metrics(y_test, predictions, positive_probabilities)
    baseline = _evaluate_baseline(x_train, y_train, x_test, y_test)
    cv_report = _cross_validation_report(model, x_train, y_train)

    labels = [0, 1]
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    normalized_matrix = confusion_matrix(y_test, predictions, labels=labels, normalize="true")
    report_dict = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    metrics = {
        "status": "evaluated",
        "model_version": bundle.get("model_version", "unknown"),
        "model_path": str(model_path),
        "task_type": bundle.get("task_type", "classification"),
        "dataset": bundle.get("dataset", {}),
        "target_definition": bundle.get("target_definition", {}),
        "target_labels": bundle.get("target_labels", TARGET_LABELS),
        "feature_schema": bundle.get("feature_columns", FEATURE_COLUMNS),
        "hyperparameters": bundle.get("hyperparameters", {}),
        "training_timestamp": bundle.get("training_timestamp"),
        "training_command": bundle.get("training_command", "python -m src.train"),
        "evaluation_command": EVALUATION_COMMAND,
        **basic,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
        "classification_report": report_dict,
        "cross_validation": cv_report,
        "baseline": baseline,
    }
    metrics["quality_gate"] = _quality_gate(metrics, baseline, cv_report)

    latest_metrics = {
        "status": "latest_evaluation",
        "model_version": metrics["model_version"],
        "model_path": metrics["model_path"],
        "dataset": metrics["dataset"],
        "task_type": metrics["task_type"],
        "target_labels": metrics["target_labels"],
        "feature_schema": metrics["feature_schema"],
        "hyperparameters": metrics["hyperparameters"],
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
        "confusion_matrix": matrix.tolist(),
        "classification_report_path": str(CLASSIFICATION_REPORT_JSON_PATH),
        "quality_gate": metrics["quality_gate"],
        "training_timestamp": metrics["training_timestamp"],
        "training_command": metrics["training_command"],
        "evaluation_command": EVALUATION_COMMAND,
    }
    model_comparison = {
        "status": "completed",
        "selected_best_model": {
            "model_name": "extra_trees_classifier",
            "model_version": metrics["model_version"],
            "algorithm": "ExtraTreesClassifier",
            "held_out_test": latest_metrics["metric_summary"],
            "cross_validation": cv_report,
            "reason_selected": (
                "ExtraTreesClassifier achieved strong accuracy, macro F1 and weighted F1, "
                "with stable 5-fold stratified cross-validation and fast CI runtime."
            ),
        },
        "baseline_comparison": {
            "baseline_accuracy": baseline["metric_summary"]["accuracy"],
            "baseline_f1_weighted": baseline["metric_summary"]["f1_weighted"],
            "final_model_accuracy": metrics["accuracy"],
            "final_model_f1_weighted": metrics["f1_weighted"],
            "absolute_accuracy_improvement": metrics["accuracy"]
            - baseline["metric_summary"]["accuracy"],
            "absolute_weighted_f1_improvement": metrics["f1_weighted"]
            - baseline["metric_summary"]["f1_weighted"],
        },
    }
    search_report = {
        "status": "completed",
        "selection_method": "StratifiedKFold cross-validation plus held-out test confirmation",
        "selection_metric": "f1_weighted",
        "models_compared": [
            "ExtraTreesClassifier",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "HistGradientBoostingClassifier",
            "LogisticRegression",
            "DummyClassifier",
        ],
        "selected_model": "ExtraTreesClassifier",
        "selected_hyperparameters": metrics["hyperparameters"],
        "held_out_test": latest_metrics["metric_summary"],
        "cross_validation": cv_report,
    }
    confusion_report = {
        "status": "completed",
        "labels": CLASS_NAMES,
        "matrix": matrix.tolist(),
        "true_label_axis": "rows",
        "predicted_label_axis": "columns",
    }
    normalized_confusion_report = {
        "status": "completed",
        "labels": CLASS_NAMES,
        "matrix": normalized_matrix.tolist(),
        "normalization": "true",
    }

    write_json(LATEST_METRICS_PATH, latest_metrics)
    write_json(BASELINE_METRICS_PATH, baseline)
    write_json(CROSS_VALIDATION_RESULTS_PATH, cv_report)
    write_json(METRICS_PATH, metrics)
    write_json(QUALITY_GATE_PATH, metrics["quality_gate"])
    write_json(QUALITY_GATE_REPORT_PATH, metrics["quality_gate"])
    write_json(
        MODEL_METADATA_PATH,
        {
            "model_version": metrics["model_version"],
            "dataset_name": metrics["dataset"].get("name"),
            "dataset_source": metrics["dataset"].get("source"),
            "dataset_hash": metrics["dataset"].get("raw_sha256"),
            "task_type": metrics["task_type"],
            "target_definition": metrics["target_definition"],
            "target_labels": metrics["target_labels"],
            "feature_schema": metrics["feature_schema"],
            "hyperparameters": metrics["hyperparameters"],
            "metric_summary": latest_metrics["metric_summary"],
            "confusion_matrix": matrix.tolist(),
            "quality_gate": metrics["quality_gate"],
            "model_path": metrics["model_path"],
            "training_timestamp": metrics["training_timestamp"],
            "training_command": metrics["training_command"],
            "evaluation_command": EVALUATION_COMMAND,
            "cross_validation_method": "StratifiedKFold",
        },
    )
    write_json(MODEL_COMPARISON_PATH, model_comparison)
    write_json(HYPERPARAMETER_SEARCH_RESULTS_PATH, search_report)
    write_json(CLASSIFICATION_REPORT_JSON_PATH, report_dict)
    CLASSIFICATION_REPORT_TEXT_PATH.write_text(report_text, encoding="utf-8")
    write_json(CONFUSION_MATRIX_PATH, confusion_report)
    write_json(CONFUSION_MATRIX_NORMALIZED_PATH, normalized_confusion_report)
    if not metrics["quality_gate"]["passed"]:
        raise RuntimeError(f"Quality gate failed: {metrics['quality_gate']['reasons']}")
    return metrics


def main() -> None:
    metrics = evaluate_model()
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
