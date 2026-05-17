"""Train and save the scikit-learn model bundle used by every serving path.

This module is responsible for turning the processed CSV into a saved model
artefact. It is run locally, in GitHub Actions, during Docker image creation,
and before Kind deployment. The saved joblib bundle contains the fitted pipeline,
feature schema, target labels, selected hyperparameters, and metadata because the
Flask API must not depend on unwritten training assumptions.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import (
    DATA_DOI,
    DATA_SHA256,
    DATA_SOURCE_PAGE,
    DATASET_NAME,
    FEATURE_COLUMNS,
    POSITIVE_CLASS_THRESHOLD,
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

MODEL_PATH = Path("models/wine_quality_classifier.joblib")
TRAIN_METADATA_PATH = Path("reports/metrics/train_metadata.json")
HYPERPARAMETER_SEARCH_RESULTS_PATH = Path("reports/metrics/hyperparameter_search_results.json")
MODEL_VERSION = get_current_version()
RANDOM_STATE = 42
TEST_SIZE = 0.2
TRAINING_COMMAND = "python -m src.train"
CATEGORICAL_FEATURES: list[str] = []
NUMERIC_FEATURES = FEATURE_COLUMNS.copy()
MODEL_HYPERPARAMETERS: dict[str, Any] = {
    "algorithm": "ExtraTreesClassifier",
    "classifier": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
    },
    "train_test_split": {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "shuffle": True,
        "stratify": TARGET_COLUMN,
    },
    "preprocessing": {
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "numeric_imputer_strategy": "median",
        "numeric_scaler": "StandardScaler",
        "column_transformer_remainder": "drop",
    },
    "selection": {
        "method": "reproducible GridSearchCV plus ensemble comparison",
        "main_scoring_metric": "f1_macro",
        "secondary_metrics": [
            "balanced_accuracy",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_weighted",
        ],
        "cross_validation": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "selection_rule": (
            "highest macro F1 unless an ensemble improves enough to justify extra runtime"
        ),
    },
}


# ==============================================================================
# Model construction
# ==============================================================================
#
# Training uses a scikit-learn Pipeline so the imputer, scaler, and classifier are
# saved together. This is important for the Flask API and Docker image because
# prediction must use the same preprocessing steps as training.


def selected_training_configuration() -> dict[str, Any]:
    """Use saved model-selection evidence when it exists, otherwise use the default.

    This is the link between `src.model_selection` and training. If the search
    report has been generated, training follows the recorded choice; otherwise a
    documented ExtraTrees default is used so a fresh checkout still works.
    """
    default_config = {
        "model_name": "extra_trees_default",
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
    if model_name == "extra_trees_tuned":
        return {
            "model_name": model_name,
            "algorithm": "ExtraTreesClassifier",
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
    """Build the selected estimator without changing the recorded hyperparameters."""
    if config["algorithm"] == "VotingClassifier":
        voting = config["classifier"].get("voting", "soft")
        extra_trees_params = config["classifier"].get(
            "extra_trees",
            MODEL_HYPERPARAMETERS["classifier"],
        )
        random_forest_params = config["classifier"].get(
            "random_forest",
            {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_leaf": 1,
                "class_weight": "balanced_subsample",
                "random_state": RANDOM_STATE,
                "n_jobs": 1,
            },
        )
        logistic_params = config["classifier"].get(
            "logistic_regression",
            {
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
            },
        )
        return VotingClassifier(
            estimators=[
                ("extra_trees", ExtraTreesClassifier(**extra_trees_params)),
                ("random_forest", RandomForestClassifier(**random_forest_params)),
                ("logistic_regression", LogisticRegression(**logistic_params)),
            ],
            voting=voting,
            n_jobs=1,
        )
    return ExtraTreesClassifier(**config["classifier"])


def build_pipeline() -> Pipeline:
    """Build the preprocessing and classifier pipeline saved as the model artefact.

    The imputer, scaler, and classifier are saved together. That design prevents
    the API, Docker container, or Kind deployment from applying different
    preprocessing to the same input features.
    """
    config = selected_training_configuration()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
    model = _selected_classifier(config)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load processed data, creating it first so a fresh checkout can train.

    This avoids a hidden setup step: if `data/processed/` is missing, the
    reproducible preprocessing stage is run rather than expecting the marker to
    create the CSV by hand.
    """
    if not path.exists():
        preprocess_dataset(output_path=path)
    return pd.read_csv(path)


def train_model(
    processed_path: Path = PROCESSED_DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train the classifier and save the model bundle used by the API and workflows.

    The same fixed, stratified split is used wherever training runs. The function
    writes `models/wine_quality_classifier.joblib`, then returns the split data
    so evaluation can use the same boundary without recomputing model behaviour.
    """
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
    # The split is fixed and stratified so local runs, Docker builds, and GitHub
    # Actions produce comparable evidence rather than a different result each time.
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    selected_config = selected_training_configuration()

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
            "positive_class_threshold": POSITIVE_CLASS_THRESHOLD,
            "negative_class_label": TARGET_LABELS[0],
            "positive_class_label": TARGET_LABELS[1],
            "description": (
                "Predict whether a red wine sample is good quality from 11 physicochemical inputs."
            ),
        },
        "classes": [int(value) for value in pipeline.named_steps["classifier"].classes_],
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
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
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "target_summary": {
            "source_quality_min": float(data[SOURCE_TARGET_COLUMN].min()),
            "source_quality_max": float(data[SOURCE_TARGET_COLUMN].max()),
            "source_quality_mean": float(data[SOURCE_TARGET_COLUMN].mean()),
            "class_distribution": {
                str(key): int(value)
                for key, value in data[TARGET_COLUMN].value_counts().sort_index().to_dict().items()
            },
        },
    }
    joblib.dump(bundle, model_path)
    return bundle, x_train, x_test, y_train, y_test


def main() -> None:
    """CLI entry point that trains the model and writes training metadata."""
    bundle, _, _, _, _ = train_model()
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
        "hyperparameters": bundle["hyperparameters"],
        "selected_model": bundle["selected_model"],
        "selected_hyperparameters": bundle["selected_hyperparameters"],
        "training_timestamp": bundle["training_timestamp"],
        "training_command": TRAINING_COMMAND,
        "training_rows": bundle["training_rows"],
        "test_rows": bundle["test_rows"],
        "target_summary": bundle["target_summary"],
    }
    write_json(TRAIN_METADATA_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
