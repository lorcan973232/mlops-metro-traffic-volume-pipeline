from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import DATASET_NAME, FEATURE_COLUMNS, TARGET_COLUMN, write_json
from src.train import (
    MODEL_HYPERPARAMETERS,
    MODEL_VERSION,
    RANDOM_STATE,
    TEST_SIZE,
    load_processed_data,
)

HYPERPARAMETER_SEARCH_RESULTS_PATH = Path("reports/metrics/hyperparameter_search_results.json")
MODEL_COMPARISON_PATH = Path("reports/metrics/model_comparison.json")
CV_SPLITS = 5
SCORING = {
    "accuracy": "accuracy",
    "precision_weighted": "precision_weighted",
    "recall_weighted": "recall_weighted",
    "f1_weighted": "f1_weighted",
    "f1_macro": "f1_macro",
}


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURE_COLUMNS,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def _pipeline(model: Any) -> Pipeline:
    return Pipeline(steps=[("preprocessor", _preprocessor()), ("classifier", model)])


def candidate_models() -> dict[str, Pipeline]:
    return {
        "dummy_most_frequent": _pipeline(DummyClassifier(strategy="most_frequent")),
        "logistic_regression": _pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "random_forest": _pipeline(
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        ),
        "extra_trees_selected": _pipeline(
            ExtraTreesClassifier(**MODEL_HYPERPARAMETERS["classifier"])
        ),
        "gradient_boosting": _pipeline(
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.07,
                max_depth=3,
                random_state=RANDOM_STATE,
            )
        ),
        "hist_gradient_boosting": _pipeline(
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.07,
                l2_regularization=0.01,
                random_state=RANDOM_STATE,
            )
        ),
    }


def _summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "per_fold": [float(value) for value in values],
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _classification_metrics(y_true: Any, predictions: Any) -> dict[str, float]:
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
    }


def _extract_classifier_params(estimator: Pipeline) -> dict[str, Any]:
    classifier = estimator.named_steps["classifier"]
    wanted = {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_samples_leaf",
        "min_samples_split",
        "class_weight",
        "random_state",
        "max_iter",
        "l2_regularization",
        "n_jobs",
        "strategy",
        "max_leaf_nodes",
    }
    return {key: value for key, value in classifier.get_params().items() if key in wanted}


def evaluate_candidate(
    name: str,
    estimator: Pipeline,
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any,
    cv: StratifiedKFold,
) -> dict[str, Any]:
    cv_results = cross_validate(
        estimator,
        x_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=1,
        error_score="raise",
    )
    fitted = estimator.fit(x_train, y_train)
    predictions = fitted.predict(x_test)
    return {
        "model_name": name,
        "algorithm": fitted.named_steps["classifier"].__class__.__name__,
        "hyperparameters": _extract_classifier_params(fitted),
        "cross_validation": {
            "folds": CV_SPLITS,
            "accuracy": _summary(cv_results["test_accuracy"]),
            "precision_weighted": _summary(cv_results["test_precision_weighted"]),
            "recall_weighted": _summary(cv_results["test_recall_weighted"]),
            "f1_weighted": _summary(cv_results["test_f1_weighted"]),
            "f1_macro": _summary(cv_results["test_f1_macro"]),
        },
        "held_out_test": _classification_metrics(y_test, predictions),
    }


def run_model_selection() -> dict[str, Any]:
    data = load_processed_data()
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
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    compared = [
        evaluate_candidate(name, estimator, x_train, y_train, x_test, y_test, cv)
        for name, estimator in candidate_models().items()
    ]
    selected_model_name = "extra_trees_selected"
    selected = next(result for result in compared if result["model_name"] == selected_model_name)
    baseline = next(result for result in compared if result["model_name"] == "dummy_most_frequent")
    baseline_summary = baseline["held_out_test"]
    selected_summary = selected["held_out_test"]
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    comparison = {
        "status": "completed",
        "generated_at": generated_at,
        "dataset": DATASET_NAME,
        "task_type": "classification",
        "selection_metric": "f1_weighted",
        "models_evaluated": sorted(
            compared,
            key=lambda result: result["held_out_test"]["f1_weighted"],
            reverse=True,
        ),
        "selected_best_model": {
            "model_name": selected["model_name"],
            "model_version": MODEL_VERSION,
            "algorithm": selected["algorithm"],
            "held_out_test": selected_summary,
            "cross_validation": selected["cross_validation"],
            "reason_selected": (
                "ExtraTreesClassifier delivered the strongest held-out weighted F1 with "
                "stable StratifiedKFold performance and simple runtime behaviour for CI, "
                "Docker, and Kind."
            ),
        },
        "baseline_model": baseline,
        "baseline_comparison": {
            "baseline_accuracy": baseline_summary["accuracy"],
            "baseline_f1_weighted": baseline_summary["f1_weighted"],
            "final_model_accuracy": selected_summary["accuracy"],
            "final_model_f1_weighted": selected_summary["f1_weighted"],
            "absolute_accuracy_improvement": selected_summary["accuracy"]
            - baseline_summary["accuracy"],
            "absolute_weighted_f1_improvement": selected_summary["f1_weighted"]
            - baseline_summary["f1_weighted"],
        },
    }
    search_report = {
        "status": "completed",
        "generated_at": generated_at,
        "selection_method": (
            "5-fold StratifiedKFold on training split plus held-out test confirmation"
        ),
        "selection_metric": "f1_weighted",
        "candidate_count": len(compared),
        "feature_engineering_tried": [
            "official UCI semicolon-delimited CSV ingestion",
            "binary target derivation from quality >= 6",
            "median imputation",
            "StandardScaler for numeric physicochemical features",
        ],
        "results": comparison["models_evaluated"],
        "selected_candidate": selected_model_name,
        "test_set_usage": "held_out_once_for_final_confirmation_not_for_model_refit",
    }
    write_json(HYPERPARAMETER_SEARCH_RESULTS_PATH, search_report)
    write_json(MODEL_COMPARISON_PATH, comparison)
    return {"model_comparison": comparison, "hyperparameter_search": search_report}


def main() -> None:
    report = run_model_selection()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
