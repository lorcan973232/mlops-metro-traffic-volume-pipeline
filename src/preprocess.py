from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import (
    FEATURE_COLUMNS,
    RAW_DATA_PATH,
    SECONDARY_TARGET_COLUMN,
    TARGET_COLUMN,
    download_dataset,
    load_raw_data,
    validate_raw_data,
    write_json,
)

PROCESSED_DATA_PATH = Path("data/processed/energy-efficiency-processed.csv")
PREPROCESS_REPORT_PATH = Path("reports/metrics/preprocessing.json")


def preprocess_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    validate_raw_data(raw_frame)
    processed = raw_frame[[*FEATURE_COLUMNS, TARGET_COLUMN, SECONDARY_TARGET_COLUMN]].copy()
    processed[["orientation", "glazing_area_distribution"]] = processed[
        ["orientation", "glazing_area_distribution"]
    ].astype(int)
    return processed


def preprocess_dataset(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not raw_path.exists():
        download_dataset(raw_path)
    processed = preprocess_frame(load_raw_data(raw_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    report = {
        "status": "processed",
        "input": str(raw_path),
        "output": str(output_path),
        "rows": int(len(processed)),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "secondary_target_column": SECONDARY_TARGET_COLUMN,
        "preprocessing_steps": [
            "validate official UCI schema",
            "rename X1-X8/Y1-Y2 columns to readable names",
            "cast categorical integer-coded fields for orientation and glazing distribution",
            "save deterministic processed CSV",
        ],
    }
    return processed, report


def main() -> None:
    _, report = preprocess_dataset()
    write_json(PREPROCESS_REPORT_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
