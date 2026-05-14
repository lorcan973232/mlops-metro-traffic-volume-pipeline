from __future__ import annotations

import pandas as pd

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, validate_raw_data
from src.preprocess import CLASS_COLUMN, map_quality_to_class, preprocess_frame


def valid_raw_frame(rows: int = 1001) -> pd.DataFrame:
    records = []
    qualities = [5, 6, 8]
    for index in range(rows):
        records.append(
            {
                "fixed_acidity": 7.0,
                "volatile_acidity": 0.27,
                "citric_acid": 0.36,
                "residual_sugar": 20.7,
                "chlorides": 0.045,
                "free_sulfur_dioxide": 45.0,
                "total_sulfur_dioxide": 170.0,
                "density": 1.001,
                "pH": 3.0,
                "sulphates": 0.45,
                "alcohol": 8.8,
                "quality": qualities[index % len(qualities)],
            }
        )
    return pd.DataFrame.from_records(records)


def test_raw_schema_validation_accepts_expected_wine_columns() -> None:
    report = validate_raw_data(valid_raw_frame())
    assert report["status"] == "valid"
    assert report["feature_columns"] == FEATURE_COLUMNS
    assert report["target_column"] == TARGET_COLUMN


def test_preprocessing_adds_deterministic_quality_classes() -> None:
    assert map_quality_to_class([3, 5, 6, 7, 9]).tolist() == [
        "low",
        "low",
        "medium",
        "high",
        "high",
    ]
    processed = preprocess_frame(valid_raw_frame())
    assert CLASS_COLUMN in processed.columns
    assert set(processed[CLASS_COLUMN]) == {"low", "medium", "high"}

