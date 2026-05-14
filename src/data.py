from __future__ import annotations

import hashlib
import json
import os
import ssl
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-white.csv"
)
DATA_SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/186/wine+quality"
DATA_DOI = "10.24432/C56S3T"
DATA_LICENSE = "Creative Commons Attribution 4.0 International"
DATA_SHA256 = "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836"

RAW_DATA_PATH = Path("data/raw/winequality-white.csv")
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
    "pH",
    "sulphates",
    "alcohol",
]
TARGET_COLUMN = "quality"
CLASS_COLUMN = "quality_class"
CLASS_LABELS = ("low", "medium", "high")


class DataQualityError(ValueError):
    """Raised when dataset ingestion or validation fails."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_to_path(url: str, destination: Path, timeout: int = 30) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        environment = os.environ.copy()
        environment["WINE_QUALITY_DATA_URL"] = url
        environment["WINE_QUALITY_DATA_OUTPUT"] = str(destination)
        environment["WINE_QUALITY_DATA_TIMEOUT"] = str(timeout)
        command = (
            "$ProgressPreference = 'SilentlyContinue'; "
            "Invoke-WebRequest "
            "-Uri $env:WINE_QUALITY_DATA_URL "
            "-OutFile $env:WINE_QUALITY_DATA_OUTPUT "
            "-TimeoutSec ([int]$env:WINE_QUALITY_DATA_TIMEOUT)"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            check=True,
            capture_output=True,
            env=environment,
            text=True,
        )
        return

    context = None
    if os.getenv("ALLOW_INSECURE_DATA_DOWNLOAD") == "1":
        context = ssl._create_unverified_context()

    with urllib.request.urlopen(url, timeout=timeout, context=context) as response:
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download_dataset(
    output_path: Path = RAW_DATA_PATH,
    force: bool = False,
    timeout: int = 30,
) -> dict[str, Any]:
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
                "fallback_used": False,
            }
        raise DataQualityError(
            f"Existing dataset hash mismatch. Expected {DATA_SHA256}, found {actual_hash}."
        )

    temporary_path = output_path.with_suffix(".tmp")
    try:
        _download_to_path(DATA_URL, temporary_path, timeout=timeout)
    except (urllib.error.URLError, subprocess.CalledProcessError, TimeoutError) as exc:
        if output_path.exists() and file_sha256(output_path) == DATA_SHA256:
            return {
                "status": "cached_after_download_error",
                "path": str(output_path),
                "sha256": DATA_SHA256,
                "source": DATA_SOURCE_PAGE,
                "fallback_used": True,
                "fallback_reason": str(exc),
            }
        stderr = getattr(exc, "stderr", "")
        detail = f"{exc}; stderr={stderr.strip()}" if stderr else str(exc)
        raise RuntimeError(
            "DATA_SOURCE_FALLBACK_USED=false; public UCI source could not be downloaded "
            f"and no verified local copy exists. Download error: {detail}"
        ) from exc

    actual_hash = file_sha256(temporary_path)
    if actual_hash != DATA_SHA256:
        temporary_path.unlink(missing_ok=True)
        raise DataQualityError(
            f"Downloaded dataset hash mismatch. Expected {DATA_SHA256}, found {actual_hash}."
        )
    temporary_path.replace(output_path)
    return {
        "status": "downloaded",
        "path": str(output_path),
        "sha256": actual_hash,
        "source": DATA_SOURCE_PAGE,
        "fallback_used": False,
        "tls_verification": os.getenv("ALLOW_INSECURE_DATA_DOWNLOAD") != "1",
    }


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [column.strip().replace(" ", "_") for column in frame.columns]
    return frame


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}. Run `python -m src.data`.")
    return normalise_columns(pd.read_csv(path, sep=";"))


def validate_raw_data(frame: pd.DataFrame, min_rows: int = 1000) -> dict[str, Any]:
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
    if not frame[TARGET_COLUMN].between(0, 10).all():
        raise DataQualityError("Quality target must be between 0 and 10.")

    return {
        "status": "valid",
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_values": missing_values,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
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
        "dataset": "UCI Wine Quality - white wine",
        "source": DATA_SOURCE_PAGE,
        "doi": DATA_DOI,
        "license": DATA_LICENSE,
        "ingestion": ingestion,
        "validation": validation,
    }
    write_json(INGESTION_REPORT_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
