"""Train and save the traffic-volume classification model bundle used by serving.

The saved bundle is the hand-off between training and the Flask API. It includes
the fitted scikit-learn pipeline, feature order, target labels, split details,
and dataset metadata used by tests, Docker, Kind, and monitoring.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import (
    CATEGORICAL_FEATURES,
    DATA_DOI,
    DATA_SHA256,
    DATA_SOURCE_PAGE,
    DATASET_NAME,
    FEATURE_COLUMNS,
    HIGH_TRAFFIC_THRESHOLD,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    SOURCE_TARGET_COLUMN,
    TARGET_COLUMN,
    TARGET_LABELS,
    TASK_TYPE,
    file_sha256,
    write_json,
)
from src.preprocess import PROCESSED_DATA_PATH, preprocess_dataset
from src.versioning import get_current_version

MODEL_PATH = Path("models/traffic_volume_classifier.joblib")
TRAIN_METADATA_PATH = Path("reports/metrics/train_metadata.json")
HYPERPARAMETER_SEARCH_RESULTS_PATH = Path("reports/metrics/hyperparameter_search_results.json")
MODEL_VERSION = get_current_version()
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
TRAINING_COMMAND = "python -m src.train"
MODEL_HYPERPARAMETERS: dict[str, Any] = {
    "algorithm": "HistGradientBoostingClassifier",
    "classifier": {
        "max_iter": 320,
        "learning_rate": 0.06,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.0,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    },
    "train_test_split": {
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
        "random_state": RANDOM_STATE,
        "shuffle": True,
        "stratify": TARGET_COLUMN,
        "final_fit_rows": "train_plus_validation_only",
        "final_test_usage": "untouched_until_final_evaluation",
    },
    "preprocessing": {
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_encoder": "OneHotEncoder(handle_unknown='ignore')",
        "numeric_features": NUMERIC_FEATURES,
        "numeric_scaler": "StandardScaler",
        "column_transformer_remainder": "drop",
        "fit_policy": "encoder fitted inside sklearn Pipeline after splitting",
    },
    "selection": {
        "method": "RandomizedSearchCV on training split plus validation candidate selection",
        "main_scoring_metric": "f1_macro",
        "secondary_metrics": [
            "balanced_accuracy",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_weighted",
            "roc_auc",
        ],
        "cross_validation": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "selection_rule": "highest validation macro F1 with stable cross-validation",
    },
}


def selected_training_configuration() -> dict[str, Any]:
    """Use saved model-selection results when they exist, otherwise use defaults."""
    default_config = {
        "model_name": "hist_gradient_boosting_default",
        "algorithm": MODEL_HYPERPARAMETERS["algorithm"],
        "classifier": MODEL_HYPERPARAMETERS["classifier"],
        "source": "src.train default hyperparameters",
    }
    if not HYPERPARAMETER_SEARCH_RESULTS_PATH.exists():
        return default_config
    try:
        search = json.loads(HYPERPARAMETER_SEARCH_RESULTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default_config
    selected = search.get("selected_model", {})
    model_name = selected.get("model_name")
    if model_name == "hist_gradient_boosting_tuned":
        return {
            "model_name": model_name,
            "algorithm": "HistGradientBoostingClassifier",
            "classifier": selected.get("hyperparameters", {}).get(
                "classifier",
                MODEL_HYPERPARAMETERS["classifier"],
            ),
            "source": str(HYPERPARAMETER_SEARCH_RESULTS_PATH),
        }
    if model_name == "soft_voting_ensemble":
        selected_hyperparameters = selected.get("hyperparameters", {})
        return {
            "model_name": model_name,
            "algorithm": "VotingClassifier",
            "classifier": selected_hyperparameters.get("classifier", selected_hyperparameters),
            "source": str(HYPERPARAMETER_SEARCH_RESULTS_PATH),
        }
    return default_config


def _selected_classifier(config: dict[str, Any]) -> Any:
    """Build the selected estimator from recorded hyperparameters."""
    if config["algorithm"] == "VotingClassifier":
        voting = config["classifier"].get("voting", "soft")
        tree_allowed = {
            "n_estimators",
            "criterion",
            "max_depth",
            "min_samples_leaf",
            "min_samples_split",
            "class_weight",
            "random_state",
            "n_jobs",
            "max_features",
            "max_leaf_nodes",
        }
        logistic_allowed = {
            "C",
            "class_weight",
            "max_iter",
            "random_state",
            "solver",
        }
        extra_trees_params = {
            key: value
            for key, value in config["classifier"]
            .get("extra_trees", MODEL_HYPERPARAMETERS["classifier"])
            .items()
            if key in tree_allowed
        }
        random_forest_params = {
            key: value
            for key, value in config["classifier"]
            .get(
                "random_forest",
                {
                    "n_estimators": 220,
                    "max_depth": None,
                    "min_samples_leaf": 1,
                    "class_weight": "balanced_subsample",
                    "random_state": RANDOM_STATE,
                    "n_jobs": 1,
                },
            )
            .items()
            if key in tree_allowed
        }
        logistic_params = {
            key: value
            for key, value in config["classifier"]
            .get(
                "logistic_regression",
                {
                    "max_iter": 1000,
                    "class_weight": "balanced",
                    "random_state": RANDOM_STATE,
                },
            )
            .items()
            if key in logistic_allowed
        }
        return VotingClassifier(
            estimators=[
                ("extra_trees", ExtraTreesClassifier(**extra_trees_params)),
                ("random_forest", RandomForestClassifier(**random_forest_params)),
                ("logistic_regression", LogisticRegression(**logistic_params)),
            ],
            voting=voting,
            n_jobs=1,
        )
    if config["algorithm"] == "ExtraTreesClassifier":
        return ExtraTreesClassifier(**config["classifier"])
    return HistGradientBoostingClassifier(**config["classifier"])


def build_pipeline() -> Pipeline:
    """Build the preprocessing and classifier pipeline used for final training."""
    config = selected_training_configuration()
    # The transformers are fitted inside the pipeline after the split. That keeps
    # preprocessing tied to training data and avoids fitting encoders or scalers
    # on the held-out test set.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", _selected_classifier(config))]
    )


def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load processed data, creating it first so a fresh checkout can train."""
    if not path.exists():
        preprocess_dataset(output_path=path)
    return pd.read_csv(path)


def split_train_validation_test(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create the fixed stratified train/validation/test split used everywhere."""
    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    # Keep the split fixed so model selection, training, and evaluation use the
    # same boundaries on every run.
    x_train_validation, x_test, y_train_validation, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y,
    )
    validation_fraction_of_remaining = VALIDATION_SIZE / (1.0 - TEST_SIZE)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=validation_fraction_of_remaining,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y_train_validation,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def split_metadata(
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Return split sizes and class distributions for the saved reports."""
    return {
        "method": "two_stage_stratified_train_validation_test_split",
        "train_size": round(1.0 - TEST_SIZE - VALIDATION_SIZE, 10),
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "shuffle": True,
        "stratify": TARGET_COLUMN,
        "final_model_fit": "train_plus_validation",
        "final_test_policy": "untouched_by_tuning_and_model_selection",
        "rows": {
            "train": int(len(y_train)),
            "validation": int(len(y_validation)),
            "test": int(len(y_test)),
            "train_plus_validation": int(len(y_train) + len(y_validation)),
        },
        "class_distribution": {
            "train": {
                str(key): int(value)
                for key, value in y_train.value_counts().sort_index().to_dict().items()
            },
            "validation": {
                str(key): int(value)
                for key, value in y_validation.value_counts().sort_index().to_dict().items()
            },
            "test": {
                str(key): int(value)
                for key, value in y_test.value_counts().sort_index().to_dict().items()
            },
        },
    }


def train_model(
    processed_path: Path = PROCESSED_DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Train the classifier and save the model bundle used by the API."""
    data = load_processed_data(processed_path)
    x_train, x_validation, x_test, y_train, y_validation, y_test = split_train_validation_test(data)
    # The final model can use training plus validation rows because model
    # selection has already happened. The test split stays untouched until
    # evaluation.
    x_train_final = pd.concat([x_train, x_validation], axis=0)
    y_train_final = pd.concat([y_train, y_validation], axis=0)
    split_report = split_metadata(y_train, y_validation, y_test)
    pipeline = build_pipeline()
    pipeline.fit(x_train_final, y_train_final)
    selected_config = selected_training_configuration()

    # Store the model together with its schema and metadata so serving can check
    # that requests still match the trained feature order.
    raw_hash = file_sha256(RAW_DATA_PATH) if RAW_DATA_PATH.exists() else DATA_SHA256
    training_timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_version": MODEL_VERSION,
        "model": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "task_type": TASK_TYPE,
        "target_labels": TARGET_LABELS,
        "dataset": {
            "name": DATASET_NAME,
            "source": DATA_SOURCE_PAGE,
            "doi": DATA_DOI,
            "raw_sha256": raw_hash,
            "processed_path": str(processed_path),
        },
        "target_definition": {
            "source_target": SOURCE_TARGET_COLUMN,
            "model_target": TARGET_COLUMN,
            "high_traffic_threshold": HIGH_TRAFFIC_THRESHOLD,
            "negative_class_label": TARGET_LABELS[0],
            "positive_class_label": TARGET_LABELS[1],
            "description": (
                "Predict whether hourly metro interstate traffic is high from weather, "
                "calendar, and recent traffic conditions."
            ),
        },
        "classes": [int(value) for value in pipeline.named_steps["classifier"].classes_],
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
        "split": split_report,
        "hyperparameters": MODEL_HYPERPARAMETERS,
        "selected_model": {
            "model_name": selected_config["model_name"],
            "algorithm": selected_config["algorithm"],
            "source": selected_config["source"],
        },
        "selected_hyperparameters": selected_config["classifier"],
        "training_timestamp": training_timestamp,
        "training_command": TRAINING_COMMAND,
        "model_path": str(model_path),
        "training_rows": int(len(x_train_final)),
        "train_rows": int(len(x_train)),
        "validation_rows": int(len(x_validation)),
        "test_rows": int(len(x_test)),
        "target_summary": {
            "class_distribution": {
                str(key): int(value)
                for key, value in data[TARGET_COLUMN].value_counts().sort_index().to_dict().items()
            },
        },
    }
    temp_model_path = model_path.with_name(f"{model_path.name}.tmp")
    if temp_model_path.exists():
        temp_model_path.unlink()
    # Write through a temporary file so a failed save does not leave a half-written
    # model bundle for Flask, Docker, or Kind to load.
    joblib.dump(bundle, temp_model_path)
    temp_model_path.replace(model_path)
    return bundle, x_train, x_validation, x_test, y_train, y_validation, y_test


def main() -> None:
    """CLI entry point that trains the model and writes training metadata."""
    parser = argparse.ArgumentParser(description="Train the traffic-volume classifier.")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run model selection before training so saved best parameters are used.",
    )
    args = parser.parse_args()
    if args.tune:
        from src.model_selection import run_model_selection

        run_model_selection()

    bundle, _, _, _, _, _, _ = train_model()
    report = {
        "status": "trained",
        "model_version": bundle["model_version"],
        "model_path": str(MODEL_PATH),
        "dataset": bundle["dataset"],
        "target_definition": bundle["target_definition"],
        "target_labels": bundle["target_labels"],
        "classes": bundle["classes"],
        "feature_columns": bundle["feature_columns"],
        "task_type": bundle["task_type"],
        "random_state": bundle["random_state"],
        "test_size": bundle["test_size"],
        "validation_size": bundle["validation_size"],
        "split": bundle["split"],
        "hyperparameters": bundle["hyperparameters"],
        "selected_model": bundle["selected_model"],
        "selected_hyperparameters": bundle["selected_hyperparameters"],
        "training_timestamp": bundle["training_timestamp"],
        "training_command": TRAINING_COMMAND,
        "training_rows": bundle["training_rows"],
        "train_rows": bundle["train_rows"],
        "validation_rows": bundle["validation_rows"],
        "test_rows": bundle["test_rows"],
        "target_summary": bundle["target_summary"],
    }
    write_json(TRAIN_METADATA_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
