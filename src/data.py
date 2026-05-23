"""Load, download, and check the UCI Metro Interstate Traffic Volume dataset.

This module is the first stage of the pipeline. It makes sure the raw CSV gzip
matches the expected schema and hash before preprocessing creates model features.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

DATASET_NAME = "UCI Metro Interstate Traffic Volume"
DATA_SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume"
DATA_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/"
    "Metro_Interstate_Traffic_Volume.csv.gz"
)
DATA_SOURCE_NOTE = "Downloaded from the official UCI Metro Interstate Traffic Volume CSV gzip file."
DATA_DOI = "10.24432/C5X60B"
DATA_LICENSE = "Creative Commons Attribution 4.0 International"
DATA_SHA256 = "0b3679ac15173f79c6dc6c5ef8a0798d806fa5c5d7f05c84a5fa711bd1b05f07"

RAW_DATA_PATH = Path("data/raw/Metro_Interstate_Traffic_Volume.csv.gz")
RAW_CSV_PATH = RAW_DATA_PATH
INGESTION_REPORT_PATH = Path("reports/metrics/data_ingestion.json")

SOURCE_TARGET_COLUMN = "traffic_volume"
TARGET_COLUMN = "high_traffic"
TASK_TYPE = "classification"
HIGH_TRAFFIC_THRESHOLD = 3800
TARGET_LABELS = {
    0: "normal traffic",
    1: "high traffic",
}

RAW_COLUMN_NAMES = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
]

FEATURE_COLUMNS = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "hour",
    "month",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "weather_main",
    "lag_1h_volume",
    "lag_24h_volume",
    "lag_168h_volume",
    "rolling_3h_volume",
    "rolling_24h_volume",
]

NUMERIC_FEATURES = [column for column in FEATURE_COLUMNS if column != "weather_main"]
CATEGORICAL_FEATURES = ["weather_main"]
WEATHER_MAIN_VALUES = [
    "Clear",
    "Clouds",
    "Drizzle",
    "Fog",
    "Haze",
    "Mist",
    "Rain",
    "Smoke",
    "Snow",
    "Squall",
    "Thunderstorm",
]


class DataQualityError(ValueError):
    """Raised when the raw data does not match the expected public dataset."""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a report file and create its parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hash used to tie reports back to the raw data file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(
    output_path: Path = RAW_DATA_PATH,
    force: bool = False,
    timeout: int = 30,
) -> dict[str, Any]:
    """Download or reuse the raw UCI gzip file after checking its SHA-256 hash."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Reusing a cached file is only safe if its hash still matches the pinned UCI
    # file. This keeps local runs, CI, Docker, and Kind tied to the same dataset.
    if output_path.exists() and not force:
        actual_hash = file_sha256(output_path)
        if actual_hash == DATA_SHA256:
            return {
                "status": "cached",
                "path": str(output_path),
                "sha256": actual_hash,
                "source": DATA_DOWNLOAD_URL,
                "fallback_used": False,
            }
        raise DataQualityError(
            f"Existing dataset hash mismatch. Expected {DATA_SHA256}, found {actual_hash}."
        )

    with urllib.request.urlopen(DATA_DOWNLOAD_URL, timeout=timeout) as response:
        content = response.read()
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != DATA_SHA256:
        raise DataQualityError(
            f"Downloaded dataset hash mismatch. Expected {DATA_SHA256}, found {actual_hash}."
        )
    output_path.write_bytes(content)
    return {
        "status": "downloaded",
        "path": str(output_path),
        "sha256": actual_hash,
        "source": DATA_DOWNLOAD_URL,
        "fallback_used": False,
    }


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the official gzip CSV and normalise holiday blanks."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}. Run `python -m src.data`.")
    # Keep only the known source columns so later schema checks are not affected
    # by accidental extra columns in a local file.
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        frame = pd.read_csv(handle)
    frame = frame[RAW_COLUMN_NAMES].copy()
    frame["holiday"] = frame["holiday"].fillna("None")
    return frame


def validate_raw_data(frame: pd.DataFrame, min_rows: int = 48000) -> dict[str, Any]:
    """Check schema, missing values, ranges, and target distribution."""
    # These checks fail before preprocessing so the model is never trained from a
    # partial or unexpected traffic dataset.
    missing_columns = [column for column in RAW_COLUMN_NAMES if column not in frame.columns]
    if missing_columns:
        raise DataQualityError(f"Missing required columns: {missing_columns}")
    if len(frame) < min_rows:
        raise DataQualityError(f"Expected at least {min_rows} rows, found {len(frame)}.")
    missing_values = int(frame[RAW_COLUMN_NAMES].isna().sum().sum())
    if missing_values:
        raise DataQualityError(f"Dataset contains {missing_values} missing values.")
    if not set(frame["weather_main"].unique()).issubset(set(WEATHER_MAIN_VALUES)):
        raise DataQualityError("Unexpected weather_main categories found.")
    if frame[SOURCE_TARGET_COLUMN].min() < 0:
        raise DataQualityError("Traffic volume must be non-negative.")

    high_traffic = (frame[SOURCE_TARGET_COLUMN] >= HIGH_TRAFFIC_THRESHOLD).astype(int)
    target_counts = high_traffic.value_counts().sort_index().to_dict()
    return {
        "status": "valid",
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_values": missing_values,
        "feature_columns": FEATURE_COLUMNS,
        "source_target_column": SOURCE_TARGET_COLUMN,
        "target_column": TARGET_COLUMN,
        "task_type": TASK_TYPE,
        "high_traffic_threshold": HIGH_TRAFFIC_THRESHOLD,
        "target_labels": TARGET_LABELS,
        "target_distribution_before_lag_drop": {
            str(key): int(value) for key, value in target_counts.items()
        },
        "weather_main_values": sorted(str(value) for value in frame["weather_main"].unique()),
        "traffic_volume_summary": {
            "min": float(frame[SOURCE_TARGET_COLUMN].min()),
            "max": float(frame[SOURCE_TARGET_COLUMN].max()),
            "mean": float(frame[SOURCE_TARGET_COLUMN].mean()),
            "median": float(frame[SOURCE_TARGET_COLUMN].median()),
        },
        "dataset_selection_reason": (
            "This public traffic-volume dataset is close in style to bike sharing: "
            "hourly demand-like counts with weather, holiday, and date/time context. "
            "It is not the bike-sharing dataset."
        ),
        "leakage_note": (
            "The target traffic_volume is not included as a feature. Lagged traffic "
            "features are shifted from previous hours only, matching a short-horizon "
            "forecasting use case where recent observed traffic is known."
        ),
    }


def main() -> None:
    ingestion = download_dataset()
    frame = load_raw_data()
    validation = validate_raw_data(frame)
    report = {
        "dataset": DATASET_NAME,
        "source": DATA_SOURCE_PAGE,
        "download_url": DATA_DOWNLOAD_URL,
        "source_note": DATA_SOURCE_NOTE,
        "doi": DATA_DOI,
        "license": DATA_LICENSE,
        "task_type": TASK_TYPE,
        "ingestion": ingestion,
        "validation": validation,
    }
    write_json(INGESTION_REPORT_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
