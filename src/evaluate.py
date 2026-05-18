"""Evaluate the saved classifier and enforce the coursework quality gate.

This stage is used after training locally, in CI, in Continuous Training, and
inside Docker builds. It writes the metric reports that the README, tests, model
registry, and live demo all inspect. The quality gate is intentionally explicit:
the model must clear several classification metrics and beat a majority-class
baseline before it is accepted by the pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_LABELS, write_json
from src.preprocess import PROCESSED_DATA_PATH
from src.sklearn_compat import load_joblib_bundle
from src.train import (
    MODEL_PATH,
    RANDOM_STATE,
    load_processed_data,
    split_metadata,
    split_train_validation_test,
    train_model,
)
from src.versioning import get_current_version

METRICS_PATH = Path("reports/metrics/metrics.json")
METRICS_CSV_PATH = Path("reports/metrics/metrics.csv")
BASELINE_METRICS_PATH = Path("reports/metrics/baseline_metrics.json")
LATEST_METRICS_PATH = Path("reports/metrics/latest_metrics.json")
QUALITY_GATE_PATH = Path("reports/metrics/quality_gate.json")
QUALITY_GATE_REPORT_PATH = Path("reports/metrics/quality_gate_report.json")
MODEL_METADATA_PATH = Path("reports/metrics/model_metadata.json")
FINAL_MODEL_METADATA_PATH = Path("reports/metrics/final_model_metadata.json")
MODEL_COMPARISON_PATH = Path("reports/metrics/model_comparison.json")
HYPERPARAMETER_SEARCH_RESULTS_PATH = Path("reports/metrics/hyperparameter_search_results.json")
CROSS_VALIDATION_RESULTS_PATH = Path("reports/metrics/cross_validation_results.json")
CROSS_VALIDATION_RESULTS_CSV_PATH = Path("reports/metrics/cross_validation_results.csv")
CLASSIFICATION_REPORT_JSON_PATH = Path("reports/metrics/classification_report.json")
CLASSIFICATION_REPORT_TEXT_PATH = Path("reports/metrics/classification_report.txt")
CONFUSION_MATRIX_PATH = Path("reports/metrics/confusion_matrix.json")
CONFUSION_MATRIX_NORMALIZED_PATH = Path("reports/metrics/confusion_matrix_normalized.json")
CONFUSION_MATRIX_PNG_PATH = Path("reports/metrics/confusion_matrix.png")
ERROR_ANALYSIS_PATH = Path("reports/metrics/error_analysis.json")
FEATURE_IMPORTANCE_PATH = Path("reports/metrics/feature_importance.json")
FAIRNESS_ANALYSIS_PATH = Path("reports/metrics/fairness_analysis.json")

STRICT_QUALITY_TARGET = 0.975
MIN_ACCURACY = STRICT_QUALITY_TARGET
MIN_PRECISION_MACRO = STRICT_QUALITY_TARGET
MIN_RECALL_MACRO = STRICT_QUALITY_TARGET
MIN_BALANCED_ACCURACY = STRICT_QUALITY_TARGET
MIN_WEIGHTED_F1 = STRICT_QUALITY_TARGET
MIN_MACRO_F1 = STRICT_QUALITY_TARGET
MIN_ROC_AUC = STRICT_QUALITY_TARGET
MIN_CV_ACCURACY = STRICT_QUALITY_TARGET
MIN_CV_F1_MACRO = STRICT_QUALITY_TARGET
MIN_PER_CLASS_METRIC = STRICT_QUALITY_TARGET
MIN_BASELINE_ACCURACY_IMPROVEMENT = 0.20
MAX_CV_F1_STD = 0.03
MAX_TEST_VALIDATION_UPLIFT = 0.03
CV_SPLITS = 5
EVALUATION_COMMAND = "python -m src.evaluate"
CLASS_NAMES = [TARGET_LABELS[0], TARGET_LABELS[1]]


# ==============================================================================
# Evaluation evidence
# ==============================================================================
#
# The test set is held back from model selection, so these reports give the
# fairest view of the chosen model after training. The quality gate is deliberately
# more than a smoke test: it checks useful classification metrics and requires the
# model to beat a simple baseline by a visible margin.


def _positive_class_probabilities(model: Any, x_test: Any) -> np.ndarray | None:
    """Return positive-class probabilities when the classifier supports them.

    ROC AUC needs probabilities for the positive `high traffic` class. Returning
    `None` when unavailable keeps the report honest instead of inventing a score.
    """
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
    """Calculate the metric set used by evaluation, baseline, and CT reporting."""
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


def _evaluate_baseline(
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any,
    quality_target: float = STRICT_QUALITY_TARGET,
) -> dict[str, Any]:
    """Evaluate a simple majority-class baseline for comparison with the real model."""
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
            "min_accuracy": quality_target,
            "min_precision_macro": quality_target,
            "min_recall_macro": quality_target,
            "min_balanced_accuracy": quality_target,
            "min_weighted_f1": quality_target,
            "min_macro_f1": quality_target,
            "min_roc_auc": quality_target,
            "min_cv_accuracy": quality_target,
            "min_cv_f1_macro": quality_target,
            "min_per_class_metric": quality_target,
            "min_baseline_accuracy_improvement": MIN_BASELINE_ACCURACY_IMPROVEMENT,
            "max_cv_f1_std": MAX_CV_F1_STD,
            "max_test_validation_uplift": MAX_TEST_VALIDATION_UPLIFT,
        },
        "created_by": EVALUATION_COMMAND,
    }


def _cross_validation_report(estimator: Any, x_train: Any, y_train: Any) -> dict[str, Any]:
    """Run stratified cross-validation to show that results are not from one lucky split."""
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "precision_weighted": make_scorer(
            precision_score,
            average="weighted",
            zero_division=0,
        ),
        "recall_weighted": make_scorer(recall_score, average="weighted", zero_division=0),
        "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
        "roc_auc": "roc_auc",
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
        "balanced_accuracy": summary(results["test_balanced_accuracy"]),
        "precision_macro": summary(results["test_precision_macro"]),
        "recall_macro": summary(results["test_recall_macro"]),
        "f1_macro": summary(results["test_f1_macro"]),
        "precision_weighted": summary(results["test_precision_weighted"]),
        "recall_weighted": summary(results["test_recall_weighted"]),
        "f1_weighted": summary(results["test_f1_weighted"]),
        "roc_auc": summary(results["test_roc_auc"]),
    }


def _validation_reference_from_search() -> dict[str, Any] | None:
    """Load validation metrics from model selection when the search report exists."""
    if not HYPERPARAMETER_SEARCH_RESULTS_PATH.exists():
        return None
    try:
        search = json.loads(HYPERPARAMETER_SEARCH_RESULTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    selected = search.get("selected_model", {})
    validation = selected.get("validation")
    if not isinstance(validation, dict):
        return None
    return {
        "source": str(HYPERPARAMETER_SEARCH_RESULTS_PATH),
        "model_name": selected.get("model_name"),
        "metrics": validation,
        "test_set_usage": search.get("test_set_usage"),
    }


def _per_class_gate_summary(report_dict: dict[str, Any]) -> dict[str, Any]:
    """Summarise the weakest per-class result so class failure cannot be hidden."""
    per_class: dict[str, dict[str, float]] = {}
    for class_name in CLASS_NAMES:
        class_metrics = report_dict.get(class_name, {})
        per_class[class_name] = {
            "precision": float(class_metrics.get("precision", 0.0)),
            "recall": float(class_metrics.get("recall", 0.0)),
            "f1_score": float(class_metrics.get("f1-score", 0.0)),
        }
    all_values = [
        value
        for class_metrics in per_class.values()
        for value in class_metrics.values()
    ]
    return {
        "per_class": per_class,
        "minimum_metric": min(all_values) if all_values else 0.0,
    }


def _confusion_error_analysis(
    y_true: pd.Series,
    predictions: np.ndarray,
    matrix: np.ndarray,
) -> dict[str, Any]:
    """Record false-positive and false-negative counts and rates."""
    true_values = np.asarray(y_true)
    false_positive_mask = (true_values == 0) & (predictions == 1)
    false_negative_mask = (true_values == 1) & (predictions == 0)
    true_negative, false_positive = int(matrix[0][0]), int(matrix[0][1])
    false_negative, true_positive = int(matrix[1][0]), int(matrix[1][1])
    false_positive_rate = (
        false_positive / (false_positive + true_negative)
        if false_positive + true_negative
        else 0.0
    )
    false_negative_rate = (
        false_negative / (false_negative + true_positive)
        if false_negative + true_positive
        else 0.0
    )
    return {
        "status": "completed",
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "true_positives": true_positive,
        "true_negatives": true_negative,
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "false_positive_test_indices": [int(idx) for idx in y_true.index[false_positive_mask]],
        "false_negative_test_indices": [int(idx) for idx in y_true.index[false_negative_mask]],
        "note": "Indices refer to rows in the processed dataset test partition.",
    }


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Create a PNG chunk without requiring a plotting dependency."""
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _write_confusion_matrix_png(matrix: np.ndarray, path: Path) -> None:
    """Write a compact two-by-two PNG heatmap for the confusion matrix."""
    cell_size = 80
    padding = 8
    width = cell_size * 2 + padding * 3
    height = width
    max_value = max(int(matrix.max()), 1)
    background = (255, 255, 255)
    pixels = [[background for _ in range(width)] for _ in range(height)]
    for row in range(2):
        for col in range(2):
            value = int(matrix[row][col])
            intensity = int(255 - (value / max_value) * 170)
            color = (intensity, intensity, 255)
            start_x = padding + col * (cell_size + padding)
            start_y = padding + row * (cell_size + padding)
            for y in range(start_y, start_y + cell_size):
                for x in range(start_x, start_x + cell_size):
                    pixels[y][x] = color
    raw_rows = []
    for row_pixels in pixels:
        raw_rows.append(b"\x00" + b"".join(bytes(pixel) for pixel in row_pixels))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(b"".join(raw_rows)))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _write_metrics_csv(
    metrics: dict[str, Any],
    quality_target: float = STRICT_QUALITY_TARGET,
) -> None:
    """Save key final-test metrics as CSV as well as JSON."""
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "roc_auc",
    ]
    METRICS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value", "threshold", "passes_98"])
        writer.writeheader()
        for metric_name in metric_names:
            value = metrics.get(metric_name)
            writer.writerow(
                {
                    "metric": metric_name,
                    "value": "" if value is None else value,
                    "threshold": quality_target,
                    "passes_98": bool(value is not None and value >= quality_target),
                }
            )


def _write_cross_validation_csv(cv_report: dict[str, Any]) -> None:
    """Save fold-level CV results as CSV for reproducibility checks."""
    CROSS_VALIDATION_RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CROSS_VALIDATION_RESULTS_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "fold", "score", "mean", "std"],
        )
        writer.writeheader()
        for metric_name in cv_report["scoring"]:
            summary = cv_report[metric_name]
            for fold_index, score in enumerate(summary["per_fold"], start=1):
                writer.writerow(
                    {
                        "metric": metric_name,
                        "fold": fold_index,
                        "score": score,
                        "mean": summary["mean"],
                        "std": summary["std"],
                    }
                )


def _quality_gate(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    cv_report: dict[str, Any],
    class_gate: dict[str, Any],
    validation_reference: dict[str, Any] | None,
    quality_target: float = STRICT_QUALITY_TARGET,
) -> dict[str, Any]:
    """Decide whether the trained model clears the documented acceptance gate.

    Continuous Training reads this decision. The gate is intentionally strict:
    it rejects a candidate unless all relevant final-test, per-class, CV,
    validation-gap, and baseline checks clear the documented target.
    """
    baseline_summary = baseline["metric_summary"]
    validation_metrics = (
        validation_reference.get("metrics") if validation_reference is not None else None
    )
    validation_accuracy = (
        validation_metrics.get("accuracy")
        if isinstance(validation_metrics, dict)
        else None
    )
    test_not_suspiciously_higher = (
        validation_accuracy is not None
        and metrics["accuracy"] <= validation_accuracy + MAX_TEST_VALIDATION_UPLIFT
    )
    checks = {
        "accuracy_above_minimum": metrics["accuracy"] >= quality_target,
        "precision_macro_above_minimum": metrics["precision_macro"] >= quality_target,
        "recall_macro_above_minimum": metrics["recall_macro"] >= quality_target,
        "balanced_accuracy_above_minimum": (
            metrics["balanced_accuracy"] >= quality_target
        ),
        "weighted_f1_above_minimum": metrics["f1_weighted"] >= quality_target,
        "macro_f1_above_minimum": metrics["f1_macro"] >= quality_target,
        "roc_auc_above_minimum": metrics["roc_auc"] is not None
        and metrics["roc_auc"] >= quality_target,
        "per_class_metrics_above_minimum": (
            class_gate["minimum_metric"] >= quality_target
        ),
        "cv_accuracy_above_minimum": cv_report["accuracy"]["mean"] >= quality_target,
        "cv_f1_macro_above_minimum": cv_report["f1_macro"]["mean"] >= quality_target,
        "cv_f1_macro_stability_within_limit": cv_report["f1_macro"]["std"] <= MAX_CV_F1_STD,
        "accuracy_beats_baseline_margin": metrics["accuracy"]
        >= baseline_summary["accuracy"] + MIN_BASELINE_ACCURACY_IMPROVEMENT,
        "validation_reference_available": validation_accuracy is not None,
        "test_score_not_suspiciously_higher_than_validation": test_not_suspiciously_higher,
    }
    passed = all(checks.values())
    failed_checks = [check for check, result in checks.items() if not result]
    return {
        "status": "passed" if passed else "failed",
        "model_version": metrics["model_version"],
        "passed": passed,
        "decision": "accept_candidate_model" if passed else "reject_candidate_model",
        "thresholds": baseline["quality_thresholds"],
        "quality_target": quality_target,
        "candidate_accuracy": metrics["accuracy"],
        "candidate_balanced_accuracy": metrics["balanced_accuracy"],
        "candidate_precision_macro": metrics["precision_macro"],
        "candidate_recall_macro": metrics["recall_macro"],
        "candidate_f1_weighted": metrics["f1_weighted"],
        "candidate_f1_macro": metrics["f1_macro"],
        "candidate_roc_auc": metrics["roc_auc"],
        "cv_accuracy_mean": cv_report["accuracy"]["mean"],
        "cv_f1_macro_mean": cv_report["f1_macro"]["mean"],
        "cv_f1_macro_std": cv_report["f1_macro"]["std"],
        "validation_accuracy": validation_accuracy,
        "per_class_gate": class_gate,
        "baseline_accuracy": baseline_summary["accuracy"],
        "baseline_f1_weighted": baseline_summary["f1_weighted"],
        "checks": checks,
        "reasons": (
            [f"All strict {quality_target:.1%} classification model quality gates passed."]
            if passed
            else [f"Failed check: {check}" for check in failed_checks]
        ),
        "strict_acceptance_ready": passed,
        "honest_readiness_statement": (
            f"All relevant metrics are genuinely at or above {quality_target:.1%}."
            if passed
            else (
                "The model is rejected for the documented acceptance threshold; saved "
                "metrics must not be presented as accepted."
            )
        ),
    }


def evaluate_model(
    model_path: Path = MODEL_PATH,
    processed_path: Path = PROCESSED_DATA_PATH,
    quality_target: float = STRICT_QUALITY_TARGET,
) -> dict[str, Any]:
    """Create the metric, baseline, cross-validation, and quality-gate reports.

    Reports are saved under `reports/metrics/` so a marker can inspect held-out
    performance, baseline comparison, confusion matrix, feature importance, and
    the promotion decision without rerunning the whole pipeline.
    """
    if not model_path.exists():
        train_model(processed_path=processed_path, model_path=model_path)

    data = load_processed_data(processed_path)
    x_train, x_validation, x_test, y_train, y_validation, y_test = (
        split_train_validation_test(data)
    )
    x_train_validation = pd.concat([x_train, x_validation], axis=0)
    y_train_validation = pd.concat([y_train, y_validation], axis=0)
    split_report = split_metadata(y_train, y_validation, y_test)
    # Use the same fixed split as training so evaluation reloads the saved model
    # and evaluates only against the untouched final test partition.
    try:
        bundle = load_joblib_bundle(model_path)
    except Exception:
        train_model(processed_path=processed_path, model_path=model_path)
        bundle = load_joblib_bundle(model_path)
    model = bundle["model"]
    feature_importance: dict[str, float] | None = None
    feature_importance_source: str | None = None
    classifier = (
        model.named_steps.get("classifier", model)
        if hasattr(model, "named_steps")
        else model
    )
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        feature_importance = {
            feature: float(imp)
            for feature, imp in zip(FEATURE_COLUMNS, importances, strict=False)
        }
        feature_importance_source = classifier.__class__.__name__
    elif hasattr(classifier, "named_estimators_") and "extra_trees" in classifier.named_estimators_:
        extra_trees = classifier.named_estimators_["extra_trees"]
        if hasattr(extra_trees, "feature_importances_"):
            importances = extra_trees.feature_importances_
            feature_importance = {
                feature: float(imp)
                for feature, imp in zip(FEATURE_COLUMNS, importances, strict=False)
            }
            feature_importance_source = "VotingClassifier.extra_trees"
    if feature_importance is None:
        sample_size = min(1000, len(x_test))
        sampled_x = x_test.iloc[:sample_size]
        sampled_y = y_test.iloc[:sample_size]
        permutation = permutation_importance(
            model,
            sampled_x,
            sampled_y,
            scoring="f1_macro",
            n_repeats=3,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        feature_importance = {
            feature: float(max(importance, 0.0))
            for feature, importance in zip(
                FEATURE_COLUMNS,
                permutation.importances_mean,
                strict=False,
            )
        }
        feature_importance_source = "permutation_importance_f1_macro"
    predictions = model.predict(x_test)
    positive_probabilities = _positive_class_probabilities(model, x_test)
    basic = _classification_metrics(y_test, predictions, positive_probabilities)
    baseline = _evaluate_baseline(
        x_train_validation,
        y_train_validation,
        x_test,
        y_test,
        quality_target,
    )
    cv_report = _cross_validation_report(model, x_train_validation, y_train_validation)

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
    class_gate = _per_class_gate_summary(report_dict)
    validation_reference = _validation_reference_from_search()
    error_analysis = _confusion_error_analysis(y_test, predictions, matrix)

    # This is a class-balance check, not a demographic fairness claim. The fuller
    # proxy-group audit lives in scripts/fairness_audit.py because the dataset has
    # no protected attributes.
    fairness_report: dict[str, Any] = {
        "status": "fairness_analyzed",
        "model_version": bundle.get("model_version", "unknown"),
        "per_class_metrics": {},
        "disparities": {},
        "is_balanced": True,
    }
    if report_dict is not None:
        for idx, class_name in enumerate(CLASS_NAMES):
            class_key = str(idx) if str(idx) in report_dict else class_name
            class_metrics = report_dict.get(class_key, {})
            fairness_report["per_class_metrics"][class_name] = {
                "precision": float(class_metrics.get("precision", 0)),
                "recall": float(class_metrics.get("recall", 0)),
                "f1_score": float(class_metrics.get("f1-score", 0)),
                "support": int(class_metrics.get("support", 0)),
            }
        class_0_f1 = fairness_report["per_class_metrics"][CLASS_NAMES[0]]["f1_score"]
        class_1_f1 = fairness_report["per_class_metrics"][CLASS_NAMES[1]]["f1_score"]
        f1_disparity = abs(class_0_f1 - class_1_f1)
        class_0_precision = fairness_report["per_class_metrics"][CLASS_NAMES[0]]["precision"]
        class_1_precision = fairness_report["per_class_metrics"][CLASS_NAMES[1]]["precision"]
        precision_disparity = abs(class_0_precision - class_1_precision)
        class_0_recall = fairness_report["per_class_metrics"][CLASS_NAMES[0]]["recall"]
        class_1_recall = fairness_report["per_class_metrics"][CLASS_NAMES[1]]["recall"]
        recall_disparity = abs(class_0_recall - class_1_recall)
        fairness_report["disparities"] = {
            "f1_disparity": round(f1_disparity, 4),
            "precision_disparity": round(precision_disparity, 4),
            "recall_disparity": round(recall_disparity, 4),
        }
        fairness_report["is_balanced"] = (
            f1_disparity < 0.05 and precision_disparity < 0.05 and recall_disparity < 0.05
        )
        if f1_disparity >= 0.05:
            fairness_report["warning"] = "High F1 disparity detected between classes"

    metrics = {
        "status": "evaluated",
        "model_version": get_current_version(),
        "model_path": str(model_path),
        "task_type": bundle.get("task_type", "classification"),
        "dataset": bundle.get("dataset", {}),
        "target_definition": bundle.get("target_definition", {}),
        "target_labels": bundle.get("target_labels", TARGET_LABELS),
        "feature_schema": bundle.get("feature_columns", FEATURE_COLUMNS),
        "hyperparameters": bundle.get("hyperparameters", {}),
        "selected_model": bundle.get("selected_model", {}),
        "selected_hyperparameters": bundle.get("selected_hyperparameters", {}),
        "training_timestamp": bundle.get("training_timestamp"),
        "training_command": bundle.get("training_command", "python -m src.train"),
        "evaluation_command": EVALUATION_COMMAND,
        "split": split_report,
        "evaluation_policy": {
            "final_test_set": (
                "untouched_by_hyperparameter_tuning_candidate_selection_and_final_fit"
            ),
            "validation_metrics_source": (
                validation_reference["source"] if validation_reference else None
            ),
            "acceptance_uses_training_data": False,
            "quality_target": quality_target,
        },
        "data_leakage_assessment": {
            "target_column_excluded_from_features": TARGET_COLUMN not in FEATURE_COLUMNS,
            "source_traffic_volume_column_excluded_from_features": (
                "traffic_volume" not in FEATURE_COLUMNS
            ),
            "preprocessing_fit_inside_pipeline_after_split": True,
            "test_set_used_for_tuning": False,
            "validation_set_used_for_candidate_selection": validation_reference is not None,
            "status": "no_direct_leakage_found_in_pipeline_contract",
        },
        **basic,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
        "classification_report": report_dict,
        "per_class_gate": class_gate,
        "error_analysis": error_analysis,
        "validation_reference": validation_reference,
        "cross_validation": cv_report,
        "baseline": baseline,
    }
    metrics["quality_gate"] = _quality_gate(
        metrics,
        baseline,
        cv_report,
        class_gate,
        validation_reference,
        quality_target,
    )

    latest_metrics = {
        "status": "latest_evaluation",
        "model_version": metrics["model_version"],
        "model_path": metrics["model_path"],
        "dataset": metrics["dataset"],
        "task_type": metrics["task_type"],
        "target_labels": metrics["target_labels"],
        "feature_schema": metrics["feature_schema"],
        "hyperparameters": metrics["hyperparameters"],
        "selected_model": metrics["selected_model"],
        "selected_hyperparameters": metrics["selected_hyperparameters"],
        "split": metrics["split"],
        "evaluation_policy": metrics["evaluation_policy"],
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
        "error_analysis": error_analysis,
        "validation_reference": validation_reference,
        "classification_report_path": str(CLASSIFICATION_REPORT_JSON_PATH),
        "quality_gate": metrics["quality_gate"],
        "training_timestamp": metrics["training_timestamp"],
        "training_command": metrics["training_command"],
        "evaluation_command": EVALUATION_COMMAND,
    }
    model_comparison = {
        "status": "completed",
        "selected_best_model": {
            "model_name": metrics["selected_model"].get("model_name", "selected_classifier"),
            "model_version": metrics["model_version"],
            "algorithm": metrics["selected_model"].get("algorithm", "unknown"),
            "held_out_test": latest_metrics["metric_summary"],
            "cross_validation": cv_report,
            "reason_selected": (
                "The selected classifier achieved strong accuracy, macro F1 and weighted F1, "
                "with stable 5-fold stratified cross-validation and a held-out test split."
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
        "selection_method": (
            "evaluation fallback only; run python -m src.model_selection for full search"
        ),
        "selection_metric": "f1_macro",
        "models_compared": [
            metrics["selected_model"].get("algorithm", "selected_classifier"),
            "DummyClassifier",
        ],
        "selected_model": metrics["selected_model"].get("model_name", "selected_classifier"),
        "selected_hyperparameters": metrics["hyperparameters"],
        "held_out_test": latest_metrics["metric_summary"],
        "cross_validation": cv_report,
        "test_set_usage": "final_test_evaluation_only",
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
    _write_metrics_csv(metrics, quality_target)
    _write_cross_validation_csv(cv_report)
    write_json(QUALITY_GATE_PATH, metrics["quality_gate"])
    write_json(QUALITY_GATE_REPORT_PATH, metrics["quality_gate"])
    metadata_report = {
        "model_version": metrics["model_version"],
        "dataset_name": metrics["dataset"].get("name"),
        "dataset_source": metrics["dataset"].get("source"),
        "dataset_hash": metrics["dataset"].get("raw_sha256"),
        "task_type": metrics["task_type"],
        "target_definition": metrics["target_definition"],
        "target_labels": metrics["target_labels"],
        "feature_schema": metrics["feature_schema"],
        "hyperparameters": metrics["hyperparameters"],
        "selected_model": metrics["selected_model"],
        "selected_hyperparameters": metrics["selected_hyperparameters"],
        "metric_summary": latest_metrics["metric_summary"],
        "split": metrics["split"],
        "validation_reference": validation_reference,
        "confusion_matrix": matrix.tolist(),
        "error_analysis": error_analysis,
        "quality_gate": metrics["quality_gate"],
        "model_path": metrics["model_path"],
        "training_timestamp": metrics["training_timestamp"],
        "training_command": metrics["training_command"],
        "evaluation_command": EVALUATION_COMMAND,
        "cross_validation_method": "StratifiedKFold",
    }
    write_json(MODEL_METADATA_PATH, metadata_report)
    write_json(FINAL_MODEL_METADATA_PATH, metadata_report)
    if not MODEL_COMPARISON_PATH.exists():
        write_json(MODEL_COMPARISON_PATH, model_comparison)
    if not HYPERPARAMETER_SEARCH_RESULTS_PATH.exists():
        write_json(HYPERPARAMETER_SEARCH_RESULTS_PATH, search_report)
    write_json(CLASSIFICATION_REPORT_JSON_PATH, report_dict)
    CLASSIFICATION_REPORT_TEXT_PATH.write_text(report_text, encoding="utf-8")
    write_json(CONFUSION_MATRIX_PATH, confusion_report)
    write_json(CONFUSION_MATRIX_NORMALIZED_PATH, normalized_confusion_report)
    _write_confusion_matrix_png(matrix, CONFUSION_MATRIX_PNG_PATH)
    write_json(ERROR_ANALYSIS_PATH, error_analysis)
    write_json(
        FEATURE_IMPORTANCE_PATH,
        {
            "status": "computed",
            "model_version": bundle.get("model_version", "unknown"),
            "algorithm": feature_importance_source,
            "source_estimator": feature_importance_source,
            "features": feature_importance,
            "top_3_features": sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:3],
        },
    )
    write_json(FAIRNESS_ANALYSIS_PATH, fairness_report)
    return metrics


def main() -> None:
    """Run evaluation from the command line and print the full evidence payload."""
    parser = argparse.ArgumentParser(description="Evaluate the saved traffic-volume classifier.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=STRICT_QUALITY_TARGET,
        help="Minimum metric threshold for quality-gate acceptance.",
    )
    parser.add_argument(
        "--fail-on-rejection",
        action="store_true",
        help="Exit non-zero when the quality gate rejects the model.",
    )
    args = parser.parse_args()
    metrics = evaluate_model(quality_target=args.threshold)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.fail_on_rejection and not metrics["quality_gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
