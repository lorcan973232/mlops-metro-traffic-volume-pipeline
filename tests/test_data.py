from __future__ import annotations

import pandas as pd

from src.data import FEATURE_COLUMNS, SECONDARY_TARGET_COLUMN, TARGET_COLUMN, validate_raw_data
from src.preprocess import preprocess_frame


def valid_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "relative_compactness": 0.98,
                "surface_area": 514.5,
                "wall_area": 294.0,
                "roof_area": 110.25,
                "overall_height": 7.0,
                "orientation": 2,
                "glazing_area": 0.0,
                "glazing_area_distribution": 0,
                "heating_load": 15.55,
                "cooling_load": 21.33,
            }
        ]
        * 700
    )


def test_raw_schema_validation_accepts_expected_energy_columns() -> None:
    report = validate_raw_data(valid_raw_frame())
    assert report["status"] == "valid"
    assert report["feature_columns"] == FEATURE_COLUMNS
    assert report["target_column"] == TARGET_COLUMN
    assert report["secondary_target_column"] == SECONDARY_TARGET_COLUMN


def test_preprocessing_keeps_simple_regression_schema() -> None:
    processed = preprocess_frame(valid_raw_frame())
    assert [*FEATURE_COLUMNS, TARGET_COLUMN, SECONDARY_TARGET_COLUMN] == list(processed.columns)
    assert processed["orientation"].dtype.kind in {"i", "u"}
    assert processed["glazing_area_distribution"].dtype.kind in {"i", "u"}
