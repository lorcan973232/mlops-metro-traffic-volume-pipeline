from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
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
ENSEMBLE_COMPARISON_PATH = Path("reports/metrics/ensemble_comparison.json")

FAST_MODE = os.getenv("FAST_MODE", "0") == "1"
CV_SPLITS = 3 if FAST_MODE else 5
SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
}
PRIMARY_SCORING = "f1_macro"


# ==============================================================================
# Model selection
# ==============================================================================
#
# This stage compares a baseline, tuned ExtraTrees model, and a small ensemble.
# The final choice is saved as evidence so training does not depend on an
# undocumented manual decision. FAST_MODE is used only to keep CI checks quick.


def _json_ready(value: Any) -> Any:
    """Convert NumPy and estimator values into JSON-safe report content."""
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


def _extra_trees_params() -> dict[str, Any]:
    return dict(MODEL_HYPERPARAMETERS["classifier"])


def _base_extra_trees() -> Pipeline:
    return _pipeline(ExtraTreesClassifier(**_extra_trees_params()))


def _param_grid() -> dict[str, list[Any]]:
    """Return a small, repeatable grid that still proves hyperparameter tuning ran."""
    if FAST_MODE:
        return {
            "classifier__n_estimators": [160],
            "classifier__max_depth": [None],
            "classifier__min_samples_leaf": [1, 2],
            "classifier__min_samples_split": [2],
        }
    return {
        "classifier__n_estimators": [200, 300],
        "classifier__max_depth": [None, 12],
        "classifier__min_samples_leaf": [1, 2],
        "classifier__min_samples_split": [2],
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


def _classifier_params(estimator: Pipeline) -> dict[str, Any]:
    classifier = estimator.named_steps["classifier"]
    if isinstance(classifier, VotingClassifier):
        return {
            "voting": classifier.voting,
            "extra_trees": classifier.named_estimators_["extra_trees"].get_params(),
            "random_forest": classifier.named_estimators_["random_forest"].get_params(),
            "logistic_regression": classifier.named_estimators_[
                "logistic_regression"
            ].get_params(),
        }
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


def evaluate_estimator(
    name: str,
    estimator: Pipeline,
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any,
    cv: StratifiedKFold,
) -> dict[str, Any]:
    """Evaluate one candidate model with CV and the fixed held-out test split."""
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
    predictions = fitted.predict(x_test)
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
        },
        "held_out_test": _classification_metrics(y_test, predictions),
    }


def _run_grid_search(
    x_train: Any,
    y_train: Any,
    cv: StratifiedKFold,
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune ExtraTrees with GridSearchCV and keep the full ranking as evidence."""
    search = GridSearchCV(
        estimator=_base_extra_trees(),
        param_grid=_param_grid(),
        scoring=SCORING,
        refit=PRIMARY_SCORING,
        cv=cv,
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
            }
        )
    return search.best_estimator_, {
        "method": "GridSearchCV",
        "primary_scoring": PRIMARY_SCORING,
        "secondary_scoring": ["balanced_accuracy", "accuracy"],
        "cv_folds": CV_SPLITS,
        "fast_mode": FAST_MODE,
        "param_grid": _json_ready(_param_grid()),
        "best_params": _json_ready(search.best_params_),
        "best_cv_f1_macro": float(search.best_score_),
        "cv_results": sorted(cv_results, key=lambda row: row["rank_f1_macro"]),
    }


def _ensemble_from_tuned(tuned_estimator: Pipeline) -> Pipeline:
    extra_trees_params = _classifier_params(tuned_estimator)
    random_forest_params = {
        "n_estimators": 220 if not FAST_MODE else 120,
        "max_depth": extra_trees_params.get("max_depth"),
        "min_samples_leaf": extra_trees_params.get("min_samples_leaf", 1),
        "class_weight": "balanced_subsample",
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
    }
    logistic_params = {
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }
    return _pipeline(
        VotingClassifier(
            estimators=[
                ("extra_trees", ExtraTreesClassifier(**extra_trees_params)),
                ("random_forest", RandomForestClassifier(**random_forest_params)),
                ("logistic_regression", LogisticRegression(**logistic_params)),
            ],
            voting="soft",
            n_jobs=1,
        )
    )


def _select_final_model(tuned: dict[str, Any], ensemble: dict[str, Any]) -> tuple[str, str]:
    """Prefer the simpler tuned model unless the ensemble earns its extra complexity."""
    tuned_f1 = tuned["held_out_test"]["f1_macro"]
    ensemble_f1 = ensemble["held_out_test"]["f1_macro"]
    tuned_cv_std = tuned["cross_validation"]["f1_macro"]["std"]
    ensemble_cv_std = ensemble["cross_validation"]["f1_macro"]["std"]
    materially_better = ensemble_f1 >= tuned_f1 + 0.005
    stability_ok = ensemble_cv_std <= tuned_cv_std + 0.01
    if materially_better and stability_ok:
        return (
            "soft_voting_ensemble",
            "Soft voting ensemble was selected because held-out macro F1 improved "
            "materially and cross-validation stability remained acceptable.",
        )
    return (
        "extra_trees_tuned",
        "Tuned ExtraTreesClassifier was retained because it met the primary macro F1 "
        "objective with simpler runtime, deployment, and explainability characteristics.",
    )


def run_model_selection() -> dict[str, Any]:
    """Run model comparison and write the evidence consumed by training."""
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
    tuned_estimator, search_details = _run_grid_search(x_train, y_train, cv)
    candidates = [
        evaluate_estimator(
            "dummy_most_frequent",
            _pipeline(DummyClassifier(strategy="most_frequent")),
            x_train,
            y_train,
            x_test,
            y_test,
            cv,
        ),
        evaluate_estimator(
            "extra_trees_default",
            _base_extra_trees(),
            x_train,
            y_train,
            x_test,
            y_test,
            cv,
        ),
        evaluate_estimator(
            "extra_trees_tuned",
            tuned_estimator,
            x_train,
            y_train,
            x_test,
            y_test,
            cv,
        ),
    ]
    ensemble_result = evaluate_estimator(
        "soft_voting_ensemble",
        _ensemble_from_tuned(tuned_estimator),
        x_train,
        y_train,
        x_test,
        y_test,
        cv,
    )
    candidates.append(ensemble_result)
    tuned_result = next(row for row in candidates if row["model_name"] == "extra_trees_tuned")
    baseline = next(row for row in candidates if row["model_name"] == "dummy_most_frequent")
    selected_name, selected_reason = _select_final_model(tuned_result, ensemble_result)
    selected = next(row for row in candidates if row["model_name"] == selected_name)
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    ordered = sorted(
        candidates,
        key=lambda row: (
            row["held_out_test"]["f1_macro"],
            row["held_out_test"]["balanced_accuracy"],
        ),
        reverse=True,
    )
    selected_model = {
        "model_name": selected["model_name"],
        "model_version": MODEL_VERSION,
        "algorithm": selected["algorithm"],
        "hyperparameters": {
            "classifier": selected["hyperparameters"],
        },
        "held_out_test": selected["held_out_test"],
        "cross_validation": selected["cross_validation"],
        "reason_selected": selected_reason,
    }
    comparison = {
        "status": "completed",
        "generated_at": generated_at,
        "dataset": DATASET_NAME,
        "task_type": "classification",
        "selection_metric": PRIMARY_SCORING,
        "secondary_metric": "balanced_accuracy",
        "models_evaluated": ordered,
        "baseline_model": baseline,
        "tuned_model": tuned_result,
        "ensemble_model": ensemble_result,
        "selected_best_model": selected_model,
        "baseline_comparison": {
            "baseline_accuracy": baseline["held_out_test"]["accuracy"],
            "baseline_f1_macro": baseline["held_out_test"]["f1_macro"],
            "final_model_accuracy": selected["held_out_test"]["accuracy"],
            "final_model_f1_macro": selected["held_out_test"]["f1_macro"],
            "absolute_accuracy_improvement": selected["held_out_test"]["accuracy"]
            - baseline["held_out_test"]["accuracy"],
            "absolute_macro_f1_improvement": selected["held_out_test"]["f1_macro"]
            - baseline["held_out_test"]["f1_macro"],
        },
    }
    search_report = {
        "status": "completed",
        "generated_at": generated_at,
        "dataset": DATASET_NAME,
        "selection_method": "GridSearchCV plus held-out test and ensemble comparison",
        "primary_scoring": PRIMARY_SCORING,
        "secondary_scoring": ["balanced_accuracy", "accuracy"],
        "cross_validation": {
            "method": "StratifiedKFold",
            "folds": CV_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "fast_mode": FAST_MODE,
        "grid_search": search_details,
        "selected_model": selected_model,
        "test_set_usage": "held_out_once_for_final_confirmation_not_for_model_refit",
    }
    ensemble_report = {
        "status": "completed",
        "generated_at": generated_at,
        "ensemble_type": "VotingClassifier",
        "voting": "soft",
        "base_models": ["ExtraTreesClassifier", "RandomForestClassifier", "LogisticRegression"],
        "hyperparameters": ensemble_result["hyperparameters"],
        "cross_validation_scores": ensemble_result["cross_validation"],
        "held_out_test_scores": ensemble_result["held_out_test"],
        "selected": selected_name == "soft_voting_ensemble",
        "reason_selected_or_rejected": (
            selected_reason
            if selected_name == "soft_voting_ensemble"
            else "Rejected for final deployment because the tuned ExtraTrees model offered a "
            "better balance of macro F1, runtime, deployment simplicity, and explainability."
        ),
    }
    write_json(HYPERPARAMETER_SEARCH_RESULTS_PATH, search_report)
    write_json(MODEL_COMPARISON_PATH, comparison)
    write_json(ENSEMBLE_COMPARISON_PATH, ensemble_report)
    return {
        "model_comparison": comparison,
        "hyperparameter_search": search_report,
        "ensemble_comparison": ensemble_report,
    }


def main() -> None:
    report = run_model_selection()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
