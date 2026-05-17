"""Tests for raw data validation and deterministic target creation."""

from __future__ import annotations

import pandas as pd

from src.data import FEATURE_COLUMNS, SOURCE_TARGET_COLUMN, TARGET_COLUMN, validate_raw_data
from src.preprocess import preprocess_frame


def valid_raw_frame() -> pd.DataFrame:
    """Build a repeated raw-data sample that satisfies the ingestion contract."""
    return pd.DataFrame(
        [
            {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "ph": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
                "quality": 5,
            },
            {
                "fixed_acidity": 7.3,
                "volatile_acidity": 0.65,
                "citric_acid": 0.0,
                "residual_sugar": 1.2,
                "chlorides": 0.065,
                "free_sulfur_dioxide": 15.0,
                "total_sulfur_dioxide": 21.0,
                "density": 0.9946,
                "ph": 3.39,
                "sulphates": 0.47,
                "alcohol": 10.0,
                "quality": 7,
            },
        ]
        * 750
    )


def test_raw_schema_validation_accepts_expected_wine_columns() -> None:
    # Ingestion should fail if the public dataset shape changes. This test checks
    # the accepted path so the expected schema is explicit and reproducible.
    report = validate_raw_data(valid_raw_frame())
    assert report["status"] == "valid"
    assert report["feature_columns"] == FEATURE_COLUMNS
    assert report["source_target_column"] == SOURCE_TARGET_COLUMN
    assert report["target_column"] == TARGET_COLUMN
    assert report["task_type"] == "classification"


def test_preprocessing_derives_binary_quality_label() -> None:
    # The assignment needs training and testing of a real model, so preprocessing
    # must produce the exact binary target used by the classifier.
    processed = preprocess_frame(valid_raw_frame())
    assert [*FEATURE_COLUMNS, SOURCE_TARGET_COLUMN, TARGET_COLUMN] == list(processed.columns)
    assert set(processed[TARGET_COLUMN].unique()) == {0, 1}
    assert processed[TARGET_COLUMN].dtype.kind in {"i", "u"}
