"""Create the model-ready Metro traffic table from the raw UCI data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data import (
    FEATURE_COLUMNS,
    HIGH_TRAFFIC_THRESHOLD,
    RAW_DATA_PATH,
    SOURCE_TARGET_COLUMN,
    TARGET_COLUMN,
    TARGET_LABELS,
    TASK_TYPE,
    download_dataset,
    load_raw_data,
    validate_raw_data,
    write_json,
)

PROCESSED_DATA_PATH = Path("data/processed/metro-traffic-processed.csv")
PREPROCESSING_REPORT_PATH = Path("reports/metrics/preprocessing.json")


def preprocess_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Create calendar, lag, rolling, and high-traffic target features.

    The raw `traffic_volume` column becomes the source for lagged features and
    the binary target. The current-hour volume is kept for reporting, but it is
    not included in `FEATURE_COLUMNS`.
    """
    validate_raw_data(frame)
    # Sorting first makes lag and rolling features meaningful. The same timestamp
    # order is used every time, so training and monitoring work from one stable
    # processed table.
    prepared = frame.sort_values("date_time").drop_duplicates(subset=["date_time"], keep="last")
    prepared = prepared.reset_index(drop=True)
    date_time = pd.to_datetime(prepared["date_time"])
    # Calendar and holiday fields give the model time context without using the
    # target traffic volume from the same hour.
    prepared["hour"] = date_time.dt.hour
    prepared["month"] = date_time.dt.month
    prepared["day_of_week"] = date_time.dt.dayofweek
    prepared["is_weekend"] = (date_time.dt.dayofweek >= 5).astype(int)
    prepared["is_holiday"] = (prepared["holiday"] != "None").astype(int)
    # Lag and rolling values are shifted before use. This avoids leaking the
    # current target value into the features.
    prepared["lag_1h_volume"] = prepared[SOURCE_TARGET_COLUMN].shift(1)
    prepared["lag_24h_volume"] = prepared[SOURCE_TARGET_COLUMN].shift(24)
    prepared["lag_168h_volume"] = prepared[SOURCE_TARGET_COLUMN].shift(168)
    prepared["rolling_3h_volume"] = prepared[SOURCE_TARGET_COLUMN].shift(1).rolling(3).mean()
    prepared["rolling_24h_volume"] = prepared[SOURCE_TARGET_COLUMN].shift(1).rolling(24).mean()
    prepared[TARGET_COLUMN] = (
        prepared[SOURCE_TARGET_COLUMN] >= HIGH_TRAFFIC_THRESHOLD
    ).astype(int)
    return prepared[[*FEATURE_COLUMNS, SOURCE_TARGET_COLUMN, TARGET_COLUMN]].dropna().reset_index(
        drop=True
    )


def preprocess_dataset(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Run deterministic preprocessing and write the processed CSV and report."""
    if not Path(raw_path).exists():
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
        "source_target_column": SOURCE_TARGET_COLUMN,
        "target_column": TARGET_COLUMN,
        "target_labels": TARGET_LABELS,
        "task_type": TASK_TYPE,
        "high_traffic_threshold": HIGH_TRAFFIC_THRESHOLD,
        "target_distribution": {
            str(key): int(value)
            for key, value in processed[TARGET_COLUMN].value_counts().sort_index().to_dict().items()
        },
        "preprocessing_steps": [
            "validate official UCI Metro Interstate Traffic Volume schema",
            "sort by date_time and keep one row per hour to avoid duplicate-hour leakage",
            "derive calendar features from date_time",
            "derive lagged traffic features using previous traffic-volume observations only",
            "derive high_traffic target where traffic_volume >= 3800",
            "save deterministic processed CSV for training, evaluation, and Flask demos",
        ],
    }
    write_json(PREPROCESSING_REPORT_PATH, report)
    return processed


def main() -> None:
    """Preprocess the raw traffic data and print the saved report."""
    preprocess_dataset()
    report = json.loads(PREPROCESSING_REPORT_PATH.read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
