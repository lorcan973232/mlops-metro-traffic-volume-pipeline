from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, validate_raw_data
from src.preprocess import CLASS_COLUMN, map_diagnosis_to_class, preprocess_frame


def valid_raw_frame(rows: int = 569) -> pd.DataFrame:
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.drop(columns=["target"]).copy()
    frame.columns = [column.strip().replace(" ", "_") for column in frame.columns]
    frame[TARGET_COLUMN] = dataset.target.astype(int)
    return frame[[*FEATURE_COLUMNS, TARGET_COLUMN]].head(rows)


def test_raw_schema_validation_accepts_expected_breast_cancer_columns() -> None:
    report = validate_raw_data(valid_raw_frame())
    assert report["status"] == "valid"
    assert report["feature_columns"] == FEATURE_COLUMNS
    assert report["target_column"] == TARGET_COLUMN
    assert report["target_mapping"] == {"0": "malignant", "1": "benign"}


def test_preprocessing_adds_deterministic_diagnosis_classes() -> None:
    assert map_diagnosis_to_class([0, 1]).tolist() == ["malignant", "benign"]
    processed = preprocess_frame(valid_raw_frame())
    assert CLASS_COLUMN in processed.columns
    assert set(processed[CLASS_COLUMN]) == {"malignant", "benign"}
