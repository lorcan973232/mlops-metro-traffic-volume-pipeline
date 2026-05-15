from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
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

MODEL_PATH = Path("models/wine_quality_classifier.joblib")
TRAIN_METADATA_PATH = Path("reports/metrics/train_metadata.json")
MODEL_VERSION = "wine-quality-extra-trees-v1"
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
        "method": "controlled scikit-learn classifier comparison for a compact public dataset",
        "main_scoring_metric": "weighted_f1",
        "secondary_metrics": ["accuracy", "macro_f1", "precision_weighted", "recall_weighted"],
        "cross_validation": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "selection_rule": (
            "highest held-out weighted F1 with stable CV performance and fast runtime"
        ),
    },
}


def build_pipeline() -> Pipeline:
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
    model = ExtraTreesClassifier(**MODEL_HYPERPARAMETERS["classifier"])
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        preprocess_dataset(output_path=path)
    return pd.read_csv(path)


def train_model(
    processed_path: Path = PROCESSED_DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

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
