from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.data import (
    CLASS_COLUMN,
    CLASS_LABELS,
    DATA_DOI,
    DATA_SHA256,
    DATA_SOURCE_PAGE,
    FEATURE_COLUMNS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TARGET_MAPPING,
    file_sha256,
    write_json,
)
from src.preprocess import PROCESSED_DATA_PATH, preprocess_dataset

MODEL_PATH = Path("models/breast_cancer_classifier.joblib")
TRAIN_METADATA_PATH = Path("reports/metrics/train_metadata.json")
MODEL_VERSION = "breast-cancer-logistic-regression-v2"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TRAINING_COMMAND = "python -m src.train"
MODEL_HYPERPARAMETERS: dict[str, Any] = {
    "algorithm": "LogisticRegression",
    "classifier": {
        "C": 0.3,
        "class_weight": "balanced",
        "max_iter": 2000,
        "penalty": "l2",
        "random_state": RANDOM_STATE,
        "solver": "lbfgs",
    },
    "train_test_split": {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratify": CLASS_COLUMN,
    },
    "preprocessing": {
        "numeric_imputer_strategy": "median",
        "numeric_scaler": "RobustScaler",
        "column_transformer_remainder": "drop",
    },
    "selection": {
        "method": (
            "dataset suitability review plus controlled scikit-learn model comparison "
            "with feature-engineering candidates"
        ),
        "main_scoring_metric": "f1_macro",
        "cross_validation": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "selection_rule": "highest held-out macro F1 with strong CV support and reproducibility",
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
                        ("scaler", RobustScaler()),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )
    model = LogisticRegression(**MODEL_HYPERPARAMETERS["classifier"])
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
    y = data[CLASS_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
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
        "class_labels": CLASS_LABELS,
        "task_type": "binary_classification",
        "dataset": {
            "name": "UCI Breast Cancer Wisconsin Diagnostic",
            "source": DATA_SOURCE_PAGE,
            "doi": DATA_DOI,
            "raw_sha256": raw_hash,
            "processed_path": str(processed_path),
        },
        "target_definition": {
            "raw_target": TARGET_COLUMN,
            "model_target": CLASS_COLUMN,
            "class_mapping": {str(key): value for key, value in TARGET_MAPPING.items()},
        },
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "hyperparameters": MODEL_HYPERPARAMETERS,
        "training_timestamp": training_timestamp,
        "training_command": TRAINING_COMMAND,
        "model_path": str(model_path),
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "class_distribution": {
            str(key): int(value) for key, value in y.value_counts().sort_index().to_dict().items()
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
        "feature_columns": bundle["feature_columns"],
        "class_labels": list(bundle["class_labels"]),
        "task_type": bundle["task_type"],
        "random_state": bundle["random_state"],
        "test_size": bundle["test_size"],
        "hyperparameters": bundle["hyperparameters"],
        "training_timestamp": bundle["training_timestamp"],
        "training_command": TRAINING_COMMAND,
        "training_rows": bundle["training_rows"],
        "test_rows": bundle["test_rows"],
        "class_distribution": bundle["class_distribution"],
    }
    write_json(TRAIN_METADATA_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
