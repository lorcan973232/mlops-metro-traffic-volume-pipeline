"""Tests for traffic data validation and deterministic target creation."""

from __future__ import annotations

import pandas as pd

from src.data import FEATURE_COLUMNS, SOURCE_TARGET_COLUMN, TARGET_COLUMN, validate_raw_data
from src.preprocess import preprocess_frame


def valid_raw_frame() -> pd.DataFrame:
    """Build a minimal raw traffic frame with the schema expected by ingestion."""
    rows = []
    for i in range(48000):
        rows.append(
            {
                "holiday": "None",
                "temp": 288.0 + (i % 10),
                "rain_1h": 0.0,
                "snow_1h": 0.0,
                "clouds_all": float(i % 100),
                "weather_main": "Clear" if i % 3 else "Clouds",
                "weather_description": "sky is clear",
                "date_time": (pd.Timestamp("2012-01-01") + pd.Timedelta(hours=i)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "traffic_volume": 3000 + (i % 2000),
            }
        )
    return pd.DataFrame(rows)


def test_raw_schema_validation_accepts_expected_traffic_columns() -> None:
    """Check raw data validation protects the expected Metro traffic schema."""
    report = validate_raw_data(valid_raw_frame())
    assert report["status"] == "valid"
    assert report["feature_columns"] == FEATURE_COLUMNS
    assert report["source_target_column"] == SOURCE_TARGET_COLUMN
    assert report["target_column"] == TARGET_COLUMN
    assert report["task_type"] == "classification"


def test_preprocessing_derives_binary_high_traffic_label() -> None:
    """Check preprocessing creates the model target and feature columns."""
    processed = preprocess_frame(valid_raw_frame())
    assert [*FEATURE_COLUMNS, SOURCE_TARGET_COLUMN, TARGET_COLUMN] == list(processed.columns)
    assert set(processed[TARGET_COLUMN].unique()) == {0, 1}
    assert processed[TARGET_COLUMN].dtype.kind in {"i", "u"}
