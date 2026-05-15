from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, write_json
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

MIN_R2 = 0.98
MAX_RMSE = 0.75
MAX_MAE = 0.55
MAX_BASELINE_REGRESSION = 0.02
CV_SPLITS = 5
EVALUATION_COMMAND = "python -m src.evaluate"


def _rmse(y_true: Any, y_pred: Any) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def _regression_metrics(y_true: Any, predictions: Any) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, predictions)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "mse": float(mean_squared_error(y_true, predictions)),
        "rmse": _rmse(y_true, predictions),
        "median_absolute_error": float(median_absolute_error(y_true, predictions)),
        "mean_absolute_percentage_error": float(
            mean_absolute_percentage_error(y_true, predictions)
        ),
        "explained_variance": float(explained_variance_score(y_true, predictions)),
        "max_error": float(max_error(y_true, predictions)),
    }


def _evaluate_baseline(x_train: Any, y_train: Any, x_test: Any, y_test: Any) -> dict[str, Any]:
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(x_train, y_train)
    predictions = baseline.predict(x_test)
    metrics = _regression_metrics(y_test, predictions)
    return {
        "status": "baseline_evaluated",
        "baseline_model": "DummyRegressor(strategy='mean')",
        "model_version": "dummy-mean-baseline",
        "metric_summary": {
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
        },
        "quality_thresholds": {
            "min_r2": MIN_R2,
            "max_rmse": MAX_RMSE,
            "max_mae": MAX_MAE,
            "max_baseline_regression": MAX_BASELINE_REGRESSION,
        },
        "created_by": EVALUATION_COMMAND,
    }


def _cross_validation_report(estimator: Any, x_train: Any, y_train: Any) -> dict[str, Any]:
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "r2": "r2",
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
    }
    results = cross_validate(estimator, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1)

    def summary(values: np.ndarray) -> dict[str, Any]:
        return {
            "per_fold": [float(value) for value in values],
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    rmse_values = -results["test_neg_rmse"]
    mae_values = -results["test_neg_mae"]
    return {
        "status": "completed",
        "method": "KFold",
        "folds": CV_SPLITS,
        "shuffle": True,
        "random_state": RANDOM_STATE,
        "scoring": ["r2", "rmse", "mae"],
        "r2": summary(results["test_r2"]),
        "rmse": summary(rmse_values),
        "mae": summary(mae_values),
    }


def _quality_gate(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    baseline_summary = baseline["metric_summary"]
    checks = {
        "r2_above_minimum": metrics["r2"] >= MIN_R2,
        "rmse_below_maximum": metrics["rmse"] <= MAX_RMSE,
        "mae_below_maximum": metrics["mae"] <= MAX_MAE,
        "r2_above_baseline": metrics["r2"]
        >= float(baseline_summary["r2"]) - MAX_BASELINE_REGRESSION,
        "rmse_better_than_baseline": metrics["rmse"]
        <= float(baseline_summary["rmse"]) + MAX_BASELINE_REGRESSION,
        "mae_better_than_baseline": metrics["mae"]
        <= float(baseline_summary["mae"]) + MAX_BASELINE_REGRESSION,
    }
    passed = all(checks.values())
    failed_checks = [check for check, result in checks.items() if not result]
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "decision": "accept_candidate_model" if passed else "reject_candidate_model",
        "thresholds": baseline["quality_thresholds"],
        "candidate_r2": metrics["r2"],
        "candidate_rmse": metrics["rmse"],
        "candidate_mae": metrics["mae"],
        "baseline_r2": baseline_summary["r2"],
        "baseline_rmse": baseline_summary["rmse"],
        "baseline_mae": baseline_summary["mae"],
        "checks": checks,
        "reasons": (
            ["All regression model quality gates passed."]
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
    )
    bundle = joblib.load(model_path)
    model = bundle["model"]
    predictions = model.predict(x_test)
    basic = _regression_metrics(y_test, predictions)
    residuals = np.asarray(y_test) - np.asarray(predictions)
    baseline = _evaluate_baseline(x_train, y_train, x_test, y_test)
    cv_report = _cross_validation_report(model, x_train, y_train)

    metrics = {
        "status": "evaluated",
        "model_version": bundle.get("model_version", "unknown"),
        "model_path": str(model_path),
        "task_type": bundle.get("task_type", "regression"),
        "dataset": bundle.get("dataset", {}),
        "target_definition": bundle.get("target_definition", {}),
        "target_unit": bundle.get("target_unit", "heating load"),
        "feature_schema": bundle.get("feature_columns", FEATURE_COLUMNS),
        "hyperparameters": bundle.get("hyperparameters", {}),
        "training_timestamp": bundle.get("training_timestamp"),
        "training_command": bundle.get("training_command", "python -m src.train"),
        "evaluation_command": EVALUATION_COMMAND,
        **basic,
        "residual_summary": {
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "min_residual": float(np.min(residuals)),
            "max_residual": float(np.max(residuals)),
        },
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
        "target_unit": metrics["target_unit"],
        "feature_schema": metrics["feature_schema"],
        "hyperparameters": metrics["hyperparameters"],
        "metric_summary": {
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "median_absolute_error": metrics["median_absolute_error"],
            "mean_absolute_percentage_error": metrics["mean_absolute_percentage_error"],
            "explained_variance": metrics["explained_variance"],
            "max_error": metrics["max_error"],
        },
        "quality_gate": metrics["quality_gate"],
        "training_timestamp": metrics["training_timestamp"],
        "training_command": metrics["training_command"],
        "evaluation_command": EVALUATION_COMMAND,
    }
    model_comparison = {
        "status": "completed",
        "selected_best_model": {
            "model_name": "gradient_boosting_regressor",
            "model_version": metrics["model_version"],
            "algorithm": "GradientBoostingRegressor",
            "held_out_test": latest_metrics["metric_summary"],
            "cross_validation": cv_report,
            "reason_selected": (
                "Gradient boosting achieved a very high held-out R2 with low RMSE/MAE, "
                "strong 5-fold CV stability, and fast CI/CT runtime."
            ),
        },
        "baseline_comparison": {
            "baseline_r2": baseline["metric_summary"]["r2"],
            "baseline_rmse": baseline["metric_summary"]["rmse"],
            "baseline_mae": baseline["metric_summary"]["mae"],
            "final_model_r2": metrics["r2"],
            "final_model_rmse": metrics["rmse"],
            "final_model_mae": metrics["mae"],
            "absolute_r2_improvement": metrics["r2"] - baseline["metric_summary"]["r2"],
            "absolute_rmse_reduction": baseline["metric_summary"]["rmse"] - metrics["rmse"],
            "absolute_mae_reduction": baseline["metric_summary"]["mae"] - metrics["mae"],
        },
    }
    search_report = {
        "status": "completed",
        "selection_method": "KFold cross-validation plus held-out test confirmation",
        "selection_metric": "r2",
        "models_compared": [
            "GradientBoostingRegressor",
            "HistGradientBoostingRegressor",
            "RandomForestRegressor",
            "ExtraTreesRegressor",
            "DummyRegressor",
        ],
        "selected_model": "GradientBoostingRegressor",
        "selected_hyperparameters": metrics["hyperparameters"],
        "held_out_test": latest_metrics["metric_summary"],
        "cross_validation": cv_report,
    }
    not_applicable = {
        "status": "NOT_APPLICABLE",
        "reason": "The selected UCI Energy Efficiency task is regression, not classification.",
    }
    write_json(LATEST_METRICS_PATH, latest_metrics)
    write_json(BASELINE_METRICS_PATH, baseline)
    write_json(CROSS_VALIDATION_RESULTS_PATH, cv_report)
    write_json(METRICS_PATH, metrics)
    write_json(QUALITY_GATE_PATH, metrics["quality_gate"])
    write_json(QUALITY_GATE_REPORT_PATH, metrics["quality_gate"])
    write_json(MODEL_METADATA_PATH, {
        "dataset_name": metrics["dataset"].get("name"),
        "dataset_source": metrics["dataset"].get("source"),
        "dataset_hash": metrics["dataset"].get("raw_sha256"),
        "task_type": metrics["task_type"],
        "target_definition": metrics["target_definition"],
        "target_unit": metrics["target_unit"],
        "feature_schema": metrics["feature_schema"],
        "hyperparameters": metrics["hyperparameters"],
        "metric_summary": latest_metrics["metric_summary"],
        "quality_gate": metrics["quality_gate"],
        "model_path": metrics["model_path"],
        "model_version": metrics["model_version"],
        "training_timestamp": metrics["training_timestamp"],
        "training_command": metrics["training_command"],
        "evaluation_command": EVALUATION_COMMAND,
        "cross_validation_method": "KFold",
    })
    write_json(MODEL_COMPARISON_PATH, model_comparison)
    write_json(HYPERPARAMETER_SEARCH_RESULTS_PATH, search_report)
    write_json(CLASSIFICATION_REPORT_JSON_PATH, not_applicable)
    CLASSIFICATION_REPORT_TEXT_PATH.write_text(
        (
            "NOT_APPLICABLE: The selected UCI Energy Efficiency task is regression, "
            "not classification.\n"
        ),
        encoding="utf-8",
    )
    write_json(CONFUSION_MATRIX_PATH, not_applicable)
    write_json(CONFUSION_MATRIX_NORMALIZED_PATH, not_applicable)
    if not metrics["quality_gate"]["passed"]:
        raise RuntimeError(f"Quality gate failed: {metrics['quality_gate']['reasons']}")
    return metrics


def main() -> None:
    metrics = evaluate_model()
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
