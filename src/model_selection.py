"""Compare traffic-volume classifiers and save model-selection reports.

This stage runs before final training. It compares candidate models on the
training and validation split, while leaving the final test split for evaluation.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import (
    CATEGORICAL_FEATURES,
    DATASET_NAME,
    NUMERIC_FEATURES,
    write_json,
)
from src.train import (
    MODEL_HYPERPARAMETERS,
    MODEL_VERSION,
    RANDOM_STATE,
    load_processed_data,
    split_metadata,
    split_train_validation_test,
)

HYPERPARAMETER_SEARCH_RESULTS_PATH = Path("reports/metrics/hyperparameter_search_results.json")
HYPERPARAMETER_RESULTS_CSV_PATH = Path("reports/metrics/hyperparameter_results.csv")
BEST_PARAMS_PATH = Path("reports/metrics/best_params.json")
MODEL_COMPARISON_PATH = Path("reports/metrics/model_comparison.json")
ENSEMBLE_COMPARISON_PATH = Path("reports/metrics/ensemble_comparison.json")

FAST_MODE = os.getenv("FAST_MODE", "0") == "1"
CV_SPLITS = 3 if FAST_MODE else 5
PRIMARY_SCORING = "f1_macro"
SEARCH_ITERATIONS = 4 if FAST_MODE else 10
SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
    "roc_auc": "roc_auc",
}


def _json_ready(value: Any) -> Any:
    """Convert NumPy values into JSON-friendly Python values for reports."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    return value


def _preprocessor() -> ColumnTransformer:
    """Build the preprocessing block shared by every candidate model."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def _pipeline(model: Any) -> Pipeline:
    """Attach one candidate classifier to the shared preprocessing block."""
    return Pipeline(steps=[("preprocessor", _preprocessor()), ("classifier", model)])


def _base_hist_gradient_boosting() -> Pipeline:
    """Return the default gradient-boosting candidate used as a strong baseline."""
    return _pipeline(HistGradientBoostingClassifier(**MODEL_HYPERPARAMETERS["classifier"]))


def _param_distributions() -> dict[str, list[Any]]:
    """Return the search space, with a smaller version for fast workflow runs."""
    if FAST_MODE:
        return {
            "classifier__max_iter": [180],
            "classifier__learning_rate": [0.08],
            "classifier__max_leaf_nodes": [31],
            "classifier__l2_regularization": [0.0],
            "classifier__class_weight": ["balanced"],
        }
    return {
        "classifier__max_iter": [180, 240, 320],
        "classifier__learning_rate": [0.04, 0.06, 0.08],
        "classifier__max_leaf_nodes": [15, 31, 63],
        "classifier__l2_regularization": [0.0, 0.01, 0.1],
        "classifier__class_weight": [None, "balanced"],
    }


def _summary(values: np.ndarray) -> dict[str, Any]:
    """Summarise fold scores while keeping the per-fold values."""
    return {
        "per_fold": [float(value) for value in values],
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _classification_metrics(
    y_true: Any,
    predictions: Any,
    positive_probabilities: np.ndarray | None,
) -> dict[str, float | None]:
    """Calculate the validation metrics used to compare candidates."""
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, predictions, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, predictions, average="weighted", zero_division=0
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
        "roc_auc": (
            float(roc_auc_score(y_true, positive_probabilities))
            if positive_probabilities is not None
            else None
        ),
    }


def _positive_class_probabilities(estimator: Pipeline, features: Any) -> np.ndarray | None:
    """Return high-traffic probabilities when the estimator supports them."""
    if not hasattr(estimator, "predict_proba"):
        return None
    probabilities = estimator.predict_proba(features)
    classes = list(estimator.classes_) if hasattr(estimator, "classes_") else [0, 1]
    if 1 not in classes:
        return None
    return probabilities[:, classes.index(1)]


def _classifier_params(estimator: Pipeline) -> dict[str, Any]:
    """Keep only readable classifier settings in the saved report."""
    classifier = estimator.named_steps["classifier"]
    wanted = {
        "max_iter",
        "learning_rate",
        "max_leaf_nodes",
        "l2_regularization",
        "class_weight",
        "random_state",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "min_samples_split",
        "max_features",
        "n_jobs",
        "strategy",
    }
    return {key: value for key, value in classifier.get_params().items() if key in wanted}


def evaluate_estimator(
    name: str,
    estimator: Pipeline,
    x_train: Any,
    y_train: Any,
    x_validation: Any,
    y_validation: Any,
    cv: StratifiedKFold,
) -> dict[str, Any]:
    """Fit one candidate and report cross-validation plus validation metrics."""
    cv_results = cross_validate(
        estimator,
        x_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=1,
        error_score="raise",
    )
    fitted = clone(estimator).fit(x_train, y_train)
    validation_predictions = fitted.predict(x_validation)
    validation_probabilities = _positive_class_probabilities(fitted, x_validation)
    return {
        "model_name": name,
        "algorithm": fitted.named_steps["classifier"].__class__.__name__,
        "hyperparameters": _json_ready(_classifier_params(fitted)),
        "cross_validation": {
            "folds": CV_SPLITS,
            "primary_metric": PRIMARY_SCORING,
            "accuracy": _summary(cv_results["test_accuracy"]),
            "balanced_accuracy": _summary(cv_results["test_balanced_accuracy"]),
            "precision_macro": _summary(cv_results["test_precision_macro"]),
            "recall_macro": _summary(cv_results["test_recall_macro"]),
            "f1_macro": _summary(cv_results["test_f1_macro"]),
            "f1_weighted": _summary(cv_results["test_f1_weighted"]),
            "roc_auc": _summary(cv_results["test_roc_auc"]),
        },
        "validation": _classification_metrics(
            y_validation,
            validation_predictions,
            validation_probabilities,
        ),
        "test_set_role": "not_evaluated_during_model_selection",
    }


def _run_hyperparameter_search(
    x_train: Any,
    y_train: Any,
    cv: StratifiedKFold,
) -> tuple[Pipeline, dict[str, Any]]:
    """Run RandomizedSearchCV for the gradient-boosting candidate."""
    search = RandomizedSearchCV(
        estimator=_base_hist_gradient_boosting(),
        param_distributions=_param_distributions(),
        n_iter=SEARCH_ITERATIONS,
        scoring=SCORING,
        refit=PRIMARY_SCORING,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    cv_results = []
    for index, params in enumerate(search.cv_results_["params"]):
        cv_results.append(
            {
                "rank_f1_macro": int(search.cv_results_["rank_test_f1_macro"][index]),
                "params": _json_ready(params),
                "mean_test_f1_macro": float(search.cv_results_["mean_test_f1_macro"][index]),
                "std_test_f1_macro": float(search.cv_results_["std_test_f1_macro"][index]),
                "mean_test_balanced_accuracy": float(
                    search.cv_results_["mean_test_balanced_accuracy"][index]
                ),
                "mean_test_accuracy": float(search.cv_results_["mean_test_accuracy"][index]),
                "mean_test_roc_auc": float(search.cv_results_["mean_test_roc_auc"][index]),
            }
        )
    return search.best_estimator_, {
        "method": "RandomizedSearchCV",
        "n_iter": SEARCH_ITERATIONS,
        "primary_scoring": PRIMARY_SCORING,
        "secondary_scoring": ["balanced_accuracy", "accuracy", "roc_auc"],
        "cv_folds": CV_SPLITS,
        "fast_mode": FAST_MODE,
        "param_distributions": _json_ready(_param_distributions()),
        "best_params": _json_ready(search.best_params_),
        "best_cv_f1_macro": float(search.best_score_),
        "cv_results": sorted(cv_results, key=lambda row: row["rank_f1_macro"]),
    }


def _write_hyperparameter_csv(rows: list[dict[str, Any]]) -> None:
    """Write the search rows as CSV for quick inspection outside JSON."""
    HYPERPARAMETER_RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank_f1_macro",
        "mean_test_f1_macro",
        "std_test_f1_macro",
        "mean_test_balanced_accuracy",
        "mean_test_accuracy",
        "mean_test_roc_auc",
        "params",
    ]
    with HYPERPARAMETER_RESULTS_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "params": json.dumps(row["params"], sort_keys=True)})


def run_model_selection() -> dict[str, Any]:
    """Compare candidates and save the reports consumed by training and README."""
    data = load_processed_data()
    x_train, x_validation, x_test, y_train, y_validation, y_test = split_train_validation_test(data)
    split_report = split_metadata(y_train, y_validation, y_test)
    # The test split is created here only so the report can record the full split.
    # Candidate selection does not score against it.
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    tuned_estimator, search_details = _run_hyperparameter_search(x_train, y_train, cv)
    candidates = [
        evaluate_estimator(
            "dummy_most_frequent",
            _pipeline(DummyClassifier(strategy="most_frequent")),
            x_train,
            y_train,
            x_validation,
            y_validation,
            cv,
        ),
        evaluate_estimator(
            "extra_trees_balanced",
            _pipeline(
                ExtraTreesClassifier(
                    n_estimators=160 if FAST_MODE else 240,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    class_weight="balanced",
                )
            ),
            x_train,
            y_train,
            x_validation,
            y_validation,
            cv,
        ),
        evaluate_estimator(
            "hist_gradient_boosting_default",
            _base_hist_gradient_boosting(),
            x_train,
            y_train,
            x_validation,
            y_validation,
            cv,
        ),
        evaluate_estimator(
            "hist_gradient_boosting_tuned",
            tuned_estimator,
            x_train,
            y_train,
            x_validation,
            y_validation,
            cv,
        ),
    ]
    selected = max(
        candidates,
        key=lambda row: (row["validation"]["f1_macro"], row["validation"]["balanced_accuracy"]),
    )
    # Select on validation macro F1, with balanced accuracy as the tie-breaker, so
    # both classes matter during model choice.
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    ordered = sorted(
        candidates,
        key=lambda row: (row["validation"]["f1_macro"], row["validation"]["balanced_accuracy"]),
        reverse=True,
    )
    selected_model = {
        "model_name": selected["model_name"],
        "model_version": MODEL_VERSION,
        "algorithm": selected["algorithm"],
        "hyperparameters": {"classifier": selected["hyperparameters"]},
        "validation": selected["validation"],
        "cross_validation": selected["cross_validation"],
        "reason_selected": (
            "Selected by highest validation macro F1; final test was not evaluated here."
        ),
    }
    baseline = next(row for row in candidates if row["model_name"] == "dummy_most_frequent")
    comparison = {
        "status": "completed",
        "generated_at": generated_at,
        "dataset": DATASET_NAME,
        "task_type": "classification",
        "selection_metric": PRIMARY_SCORING,
        "secondary_metric": "balanced_accuracy",
        "split": split_report,
        "models_evaluated": ordered,
        "baseline_model": baseline,
        "selected_best_model": selected_model,
        "baseline_comparison": {
            "baseline_validation_accuracy": baseline["validation"]["accuracy"],
            "baseline_validation_f1_macro": baseline["validation"]["f1_macro"],
            "selected_validation_accuracy": selected["validation"]["accuracy"],
            "selected_validation_f1_macro": selected["validation"]["f1_macro"],
        },
        "test_set_usage": "not_used_in_model_selection",
    }
    search_report = {
        "status": "completed",
        "generated_at": generated_at,
        "dataset": DATASET_NAME,
        "selection_method": "RandomizedSearchCV plus validation-selected candidate comparison",
        "primary_scoring": PRIMARY_SCORING,
        "secondary_scoring": ["balanced_accuracy", "accuracy", "roc_auc"],
        "cross_validation": {
            "method": "StratifiedKFold",
            "folds": CV_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "split": split_report,
        "fast_mode": FAST_MODE,
        "grid_search": search_details,
        "selected_model": selected_model,
        "test_set_usage": "not_used_in_hyperparameter_search_or_candidate_selection",
    }
    ensemble_report = {
        "status": "not_applicable",
        "reason": (
            "No ensemble selected; tuned gradient boosting gave the best validation metrics."
        ),
    }
    write_json(HYPERPARAMETER_SEARCH_RESULTS_PATH, search_report)
    _write_hyperparameter_csv(search_details["cv_results"])
    write_json(
        BEST_PARAMS_PATH,
        {
            "status": "completed",
            "generated_at": generated_at,
            "method": search_details["method"],
            "primary_scoring": PRIMARY_SCORING,
            "best_params": search_details["best_params"],
            "best_cv_f1_macro": search_details["best_cv_f1_macro"],
            "selected_model": selected_model,
        },
    )
    write_json(MODEL_COMPARISON_PATH, comparison)
    write_json(ENSEMBLE_COMPARISON_PATH, ensemble_report)
    return {
        "model_comparison": comparison,
        "hyperparameter_search": search_report,
        "ensemble_comparison": ensemble_report,
    }


def main() -> None:
    """Run model selection from the command line and print the report paths."""
    report = run_model_selection()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
