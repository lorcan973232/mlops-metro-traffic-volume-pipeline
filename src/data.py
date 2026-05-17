"""Download and validate the fixed UCI red wine dataset used by the pipeline.

This module is the first stage in the artefact. It is used by the student
locally, by GitHub Actions, by Docker build, and by later training scripts. The
important point is not just that the CSV exists; the hash, schema, target source,
and feature ranges are checked so every later report can be traced back to the
same public dataset.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

DATASET_NAME = "UCI Wine Quality - Red Wine"
DATA_SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/186/wine+quality"
DATA_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
)
DATA_SOURCE_NOTE = "Downloaded from the official UCI winequality-red.csv file."
DATA_DOI = "10.24432/C56S3T"
DATA_LICENSE = "Creative Commons Attribution 4.0 International"
DATA_SHA256 = "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e"

# ==============================================================================
# Dataset contract
# ==============================================================================
#
# The dataset is small enough for repeatable coursework runs, but the contract is
# still explicit: source URL, expected hash, schema, target rule, and reasonable
# feature ranges. These checks stop the pipeline from quietly training on the
# wrong file if the local cache is changed.

RAW_DATA_PATH = Path("data/raw/winequality-red.csv")
RAW_CSV_PATH = RAW_DATA_PATH
INGESTION_REPORT_PATH = Path("reports/metrics/data_ingestion.json")

FEATURE_COLUMNS = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]
SOURCE_TARGET_COLUMN = "quality"
TARGET_COLUMN = "quality_label"
TASK_TYPE = "classification"
POSITIVE_CLASS_THRESHOLD = 6
TARGET_LABELS = {
    0: "standard quality",
    1: "good quality",
}

RAW_COLUMN_NAMES = [*FEATURE_COLUMNS, SOURCE_TARGET_COLUMN]

FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "fixed_acidity": (4.0, 16.0),
    "volatile_acidity": (0.1, 1.6),
    "citric_acid": (0.0, 1.1),
    "residual_sugar": (0.5, 16.0),
    "chlorides": (0.01, 0.7),
    "free_sulfur_dioxide": (1.0, 80.0),
    "total_sulfur_dioxide": (5.0, 300.0),
    "density": (0.98, 1.01),
    "ph": (2.5, 4.2),
    "sulphates": (0.2, 2.2),
    "alcohol": (8.0, 16.0),
}


class DataQualityError(ValueError):
    """Raised when the raw data does not match the expected coursework dataset."""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write evidence JSON under `reports/` so workflow outputs are inspectable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    """Return the file hash used to prove that the raw dataset is reproducible."""
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
    """Download or reuse the raw UCI file after checking its SHA-256 hash.

    The GitHub runner, Docker build, and local student machine all call this
    path. Reusing the cached file is allowed only when the hash matches the known
    UCI file, which avoids a hidden manual data swap changing the model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    """Load the semicolon-delimited UCI CSV with the feature names used downstream.

    UCI publishes this file with semicolon separators. Naming the columns here
    gives preprocessing, tests, and API schema checks one shared vocabulary.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}. Run `python -m src.data`.")
    frame = pd.read_csv(path, sep=";")
    frame.columns = RAW_COLUMN_NAMES
    return frame


def validate_raw_data(frame: pd.DataFrame, min_rows: int = 1500) -> dict[str, Any]:
    """Check schema, missing values, ranges, and target distribution before training.

    This validation runs before preprocessing and training. If it fails, the
    student, marker, and CI runner know the problem is the input dataset rather
    than a later model issue.
    """
    required_columns = [*FEATURE_COLUMNS, SOURCE_TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise DataQualityError(f"Missing required columns: {missing_columns}")
    if len(frame) < min_rows:
        raise DataQualityError(f"Expected at least {min_rows} rows, found {len(frame)}.")
    non_numeric = [
        column for column in required_columns if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise DataQualityError(f"Columns must be numeric: {non_numeric}")
    missing_values = int(frame[required_columns].isna().sum().sum())
    if missing_values:
        raise DataQualityError(f"Dataset contains {missing_values} missing values.")

    invalid_ranges: dict[str, dict[str, float]] = {}
    for column, (minimum, maximum) in FEATURE_RANGES.items():
        observed_min = float(frame[column].min())
        observed_max = float(frame[column].max())
        if observed_min < minimum or observed_max > maximum:
            invalid_ranges[column] = {
                "expected_min": minimum,
                "expected_max": maximum,
                "observed_min": observed_min,
                "observed_max": observed_max,
            }
    if invalid_ranges:
        raise DataQualityError(f"Feature ranges are outside expected UCI bounds: {invalid_ranges}")

    observed_quality = sorted(int(value) for value in frame[SOURCE_TARGET_COLUMN].unique())
    if min(observed_quality) < 0 or max(observed_quality) > 10:
        raise DataQualityError(
            f"Quality scores are outside expected 0-10 range: {observed_quality}"
        )

    target_counts = (
        (frame[SOURCE_TARGET_COLUMN] >= POSITIVE_CLASS_THRESHOLD)
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    return {
        "status": "valid",
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_values": missing_values,
        "feature_columns": FEATURE_COLUMNS,
        "source_target_column": SOURCE_TARGET_COLUMN,
        "target_column": TARGET_COLUMN,
        "task_type": TASK_TYPE,
        "positive_class_threshold": POSITIVE_CLASS_THRESHOLD,
        "target_labels": TARGET_LABELS,
        "quality_scores_observed": observed_quality,
        "target_distribution": {str(key): int(value) for key, value in target_counts.items()},
        "feature_ranges": {
            column: {"min": minimum, "max": maximum}
            for column, (minimum, maximum) in FEATURE_RANGES.items()
        },
        "source_target_summary": {
            "min": float(frame[SOURCE_TARGET_COLUMN].min()),
            "max": float(frame[SOURCE_TARGET_COLUMN].max()),
            "mean": float(frame[SOURCE_TARGET_COLUMN].mean()),
        },
    }


def main() -> None:
    """Create the ingestion report used by README evidence and data workflows."""
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
