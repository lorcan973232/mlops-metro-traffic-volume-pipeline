from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import load_breast_cancer

DATA_SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic"
DATA_SOURCE_NOTE = "Loaded reproducibly from sklearn.datasets.load_breast_cancer."
DATA_DOI = "10.24432/C5DW2B"
DATA_LICENSE = "Creative Commons Attribution 4.0 International"
DATA_SHA256 = "43e012951b5fc04c166ef445da184051646d28bc4c6ba34f44fa7a4c1d656a11"

RAW_DATA_PATH = Path("data/raw/breast-cancer-wisconsin-diagnostic.csv")
INGESTION_REPORT_PATH = Path("reports/metrics/data_ingestion.json")

FEATURE_COLUMNS = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",
    "radius_error",
    "texture_error",
    "perimeter_error",
    "area_error",
    "smoothness_error",
    "compactness_error",
    "concavity_error",
    "concave_points_error",
    "symmetry_error",
    "fractal_dimension_error",
    "worst_radius",
    "worst_texture",
    "worst_perimeter",
    "worst_area",
    "worst_smoothness",
    "worst_compactness",
    "worst_concavity",
    "worst_concave_points",
    "worst_symmetry",
    "worst_fractal_dimension",
]
TARGET_COLUMN = "diagnosis"
CLASS_COLUMN = "diagnosis_class"
CLASS_LABELS = ("malignant", "benign")
TARGET_MAPPING = {0: "malignant", 1: "benign"}


class DataQualityError(ValueError):
    """Raised when dataset ingestion or validation fails."""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sklearn_frame() -> pd.DataFrame:
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.drop(columns=["target"]).copy()
    frame.columns = [column.strip().replace(" ", "_") for column in frame.columns]
    frame[TARGET_COLUMN] = dataset.target.astype(int)
    return frame[[*FEATURE_COLUMNS, TARGET_COLUMN]]


def download_dataset(
    output_path: Path = RAW_DATA_PATH,
    force: bool = False,
    timeout: int = 30,
) -> dict[str, Any]:
    del timeout
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        actual_hash = file_sha256(output_path)
        if actual_hash == DATA_SHA256:
            return {
                "status": "cached",
                "path": str(output_path),
                "sha256": actual_hash,
                "source": DATA_SOURCE_PAGE,
                "source_note": DATA_SOURCE_NOTE,
                "fallback_used": False,
            }
        raise DataQualityError(
            f"Existing dataset hash mismatch. Expected {DATA_SHA256}, found {actual_hash}."
        )

    frame = _load_sklearn_frame()
    csv_payload = frame.to_csv(index=False, lineterminator="\n")
    actual_hash = hashlib.sha256(csv_payload.encode("utf-8")).hexdigest()
    if actual_hash != DATA_SHA256:
        raise DataQualityError(
            f"Generated dataset hash mismatch. Expected {DATA_SHA256}, found {actual_hash}."
        )
    output_path.write_text(csv_payload, encoding="utf-8", newline="\n")
    return {
        "status": "generated_from_sklearn_loader",
        "path": str(output_path),
        "sha256": actual_hash,
        "source": DATA_SOURCE_PAGE,
        "source_note": DATA_SOURCE_NOTE,
        "fallback_used": False,
    }


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}. Run `python -m src.data`.")
    return pd.read_csv(path)


def validate_raw_data(frame: pd.DataFrame, min_rows: int = 500) -> dict[str, Any]:
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
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
    valid_targets = set(TARGET_MAPPING)
    actual_targets = set(frame[TARGET_COLUMN].astype(int).unique())
    if actual_targets != valid_targets:
        raise DataQualityError(
            "Diagnosis target must contain "
            f"{sorted(valid_targets)}, found {sorted(actual_targets)}."
        )

    return {
        "status": "valid",
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_values": missing_values,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "target_mapping": {str(key): value for key, value in TARGET_MAPPING.items()},
        "target_distribution": {
            str(key): int(value)
            for key, value in frame[TARGET_COLUMN].value_counts().sort_index().to_dict().items()
        },
    }


def main() -> None:
    ingestion = download_dataset()
    frame = load_raw_data()
    validation = validate_raw_data(frame)
    report = {
        "dataset": "UCI Breast Cancer Wisconsin Diagnostic",
        "source": DATA_SOURCE_PAGE,
        "source_note": DATA_SOURCE_NOTE,
        "doi": DATA_DOI,
        "license": DATA_LICENSE,
        "ingestion": ingestion,
        "validation": validation,
    }
    write_json(INGESTION_REPORT_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
