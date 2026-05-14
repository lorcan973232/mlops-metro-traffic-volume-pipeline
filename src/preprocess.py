from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import (
    CLASS_COLUMN,
    FEATURE_COLUMNS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TARGET_MAPPING,
    download_dataset,
    load_raw_data,
    validate_raw_data,
    write_json,
)

PROCESSED_DATA_PATH = Path("data/processed/breast-cancer-wisconsin-diagnostic-processed.csv")
PREPROCESS_REPORT_PATH = Path("reports/metrics/preprocessing.json")


def map_diagnosis_to_class(values: pd.Series | list[int] | int | float) -> pd.Series | str:
    scalar_input = np.isscalar(values)
    series = pd.Series([values] if scalar_input else values)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError("Diagnosis values must be numeric.")
    numeric = numeric.astype(int)
    unknown = sorted(set(numeric.unique()) - set(TARGET_MAPPING))
    if unknown:
        raise ValueError(f"Diagnosis values must map to malignant or benign: {unknown}.")
    labels = numeric.map(TARGET_MAPPING)
    if scalar_input:
        return str(labels.iloc[0])
    return pd.Series(labels.to_numpy(), index=series.index, name=CLASS_COLUMN)


def preprocess_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    validate_raw_data(raw_frame)
    processed = raw_frame[[*FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    processed[CLASS_COLUMN] = map_diagnosis_to_class(processed[TARGET_COLUMN])
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
        "class_column": CLASS_COLUMN,
        "class_mapping": {str(key): value for key, value in TARGET_MAPPING.items()},
        "class_distribution": {
            str(key): int(value)
            for key, value in processed[CLASS_COLUMN].value_counts().sort_index().to_dict().items()
        },
    }
    return processed, report


def main() -> None:
    _, report = preprocess_dataset()
    write_json(PREPROCESS_REPORT_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
