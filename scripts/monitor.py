"""Generate repeatable monitoring reports for the student project.

The project has no live production telemetry, so this script is honest about its
scope. It runs offline data-quality checks on the processed public dataset and,
when an API URL is provided, checks the deployed Flask service through `/health`
and `/predict`. Outputs are written under `reports/monitoring/` for GitHub
Actions, the README, and the live demo.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import PredictionRequestExample
from src.data import (
    DATA_SOURCE_PAGE,
    FEATURE_COLUMNS,
    SOURCE_TARGET_COLUMN,
    TARGET_COLUMN,
    write_json,
)
from src.model_registry import MODEL_METADATA_PATH, load_model_metadata
from src.preprocess import PROCESSED_DATA_PATH, preprocess_dataset

MONITORING_REPORT_PATH = Path("reports/monitoring/monitoring_report.json")
API_MONITORING_REPORT_PATH = Path("reports/monitoring/api_monitoring_report.json")
DATA_QUALITY_REPORT_PATH = Path("reports/monitoring/data_quality_report.json")


# ==============================================================================
# Continuous monitoring evidence
# ==============================================================================
#
# This is a lightweight monitoring stage rather than full production telemetry.
# It still gives repeatable reports for schema checks, missing-value checks,
# feature summaries, and optional API health/prediction checks.


def utc_now() -> str:
    """Return a UTC timestamp so monitoring reports can be compared by run."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def feature_summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Summarise each model feature for the data-quality report."""
    summary: dict[str, dict[str, float]] = {}
    for column in FEATURE_COLUMNS:
        if pd.api.types.is_numeric_dtype(frame[column]):
            summary[column] = {
                "min": float(frame[column].min()),
                "max": float(frame[column].max()),
                "mean": float(frame[column].mean()),
                "std": float(frame[column].std()),
            }
        else:
            counts = frame[column].value_counts().to_dict()
            summary[column] = {
                "unique_values": float(frame[column].nunique()),
                "missing": float(frame[column].isna().sum()),
                "top_frequency": float(max(counts.values()) if counts else 0),
            }
    return summary


def validate_monitoring_schema(frame: pd.DataFrame) -> dict[str, Any]:
    """Check that the monitoring batch still matches the training feature schema."""
    expected_columns = {*FEATURE_COLUMNS, SOURCE_TARGET_COLUMN, TARGET_COLUMN}
    missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    unexpected_columns = [column for column in frame.columns if column not in expected_columns]
    non_numeric_columns = []
    missing_values = int(frame[FEATURE_COLUMNS].isna().sum().sum()) if not missing_columns else None
    missing_by_feature = (
        {column: int(frame[column].isna().sum()) for column in FEATURE_COLUMNS}
        if not missing_columns
        else {}
    )
    checks_passed = (
        not missing_columns
        and not unexpected_columns
        and not non_numeric_columns
        and missing_values == 0
    )
    return {
        "status": "passed" if checks_passed else "failed",
        "expected_feature_schema": FEATURE_COLUMNS,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "non_numeric_columns": non_numeric_columns,
        "missing_values": missing_values,
        "missing_by_feature": missing_by_feature,
        "row_count": int(len(frame)),
    }


def load_metadata() -> dict[str, Any]:
    """Load model metadata, or return an explicit unavailable state for fresh runs."""
    if MODEL_METADATA_PATH.exists():
        return load_model_metadata()
    return {
        "model_version": "metadata_unavailable",
        "dataset_name": "UCI Metro Interstate Traffic Volume",
        "dataset_source": DATA_SOURCE_PAGE,
        "feature_schema": FEATURE_COLUMNS,
        "model_path": "models/traffic_volume_classifier.joblib",
        "metric_summary": {},
        "quality_gate": {"passed": None},
    }


def offline_monitor(processed_path: Path = PROCESSED_DATA_PATH) -> dict[str, Any]:
    """Run deterministic offline monitoring on the processed public dataset."""
    if not processed_path.exists():
        preprocess_dataset(output_path=processed_path)
    frame = pd.read_csv(processed_path)
    metadata = load_metadata()
    data_quality = validate_monitoring_schema(frame)
    data_quality_report = {
        "status": data_quality["status"],
        "timestamp_utc": utc_now(),
        "monitoring_mode": "offline_simulated",
        "reference_data_source": str(processed_path),
        "schema_validation": data_quality,
        "missing_value_checks": {
            "total_missing_values": data_quality["missing_values"],
            "missing_by_feature": data_quality["missing_by_feature"],
        },
        "feature_distribution_checks": feature_summary(frame),
        "computed_from_data": True,
    }
    retraining_required = data_quality["status"] != "passed"
    report = {
        "status": "monitored",
        "timestamp_utc": utc_now(),
        "monitoring_mode": "offline_simulated",
        "production_claim": "simulated_only",
        "production_limitation": (
            "No live production telemetry is available in this student artefact; monitoring "
            "uses the selected public dataset schema and deterministic batch checks."
        ),
        "dataset_name": metadata.get("dataset_name", "UCI Metro Interstate Traffic Volume"),
        "dataset_source": metadata.get("dataset_source", DATA_SOURCE_PAGE),
        "model_version": metadata.get("model_version"),
        "model_path": metadata.get("model_path"),
        "feature_schema": FEATURE_COLUMNS,
        "prediction_request_schema": {"features": FEATURE_COLUMNS},
        "response_status": "simulated_batch_available",
        "rows": int(len(frame)),
        "data_quality": data_quality,
        "data_quality_report_path": str(DATA_QUALITY_REPORT_PATH),
        "feature_summary": feature_summary(frame),
        "target_summary": {
            "target": TARGET_COLUMN,
            "source_target": SOURCE_TARGET_COLUMN,
            "source_traffic_volume_mean": float(frame[SOURCE_TARGET_COLUMN].mean()),
            "min": float(frame[TARGET_COLUMN].min()),
            "max": float(frame[TARGET_COLUMN].max()),
            "mean": float(frame[TARGET_COLUMN].mean()),
        },
        "retraining_required": retraining_required,
        "retraining_recommended": retraining_required,
        "reason": (
            "Investigate retraining because data-quality checks failed."
            if retraining_required
            else "No retraining required from offline schema and data-quality checks."
        ),
    }
    write_json(DATA_QUALITY_REPORT_PATH, data_quality_report)
    write_json(MONITORING_REPORT_PATH, report)
    return report


def _request_json(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Call the deployed API in a way that works from both Linux and Windows."""
    if os.name == "nt":
        return _request_json_with_powershell(url, method=method, payload=payload)
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        return int(response.status), json.loads(response.read().decode("utf-8"))


def _request_json_with_powershell(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    environment = os.environ.copy()
    environment["MONITOR_URL"] = url
    environment["MONITOR_METHOD"] = method
    environment["MONITOR_BODY"] = "" if payload is None else json.dumps(payload)
    command = r"""
$ProgressPreference = 'SilentlyContinue'
$method = $env:MONITOR_METHOD
$uri = $env:MONITOR_URL
try {
  if ($method -eq 'POST') {
    $response = Invoke-RestMethod `
      -Uri $uri `
      -Method Post `
      -ContentType 'application/json' `
      -Body $env:MONITOR_BODY `
      -TimeoutSec 10
  } else {
    $response = Invoke-RestMethod -Uri $uri -Method Get -TimeoutSec 10
  }
  $body = $response | ConvertTo-Json -Depth 10 -Compress
  [pscustomobject]@{ status_code = 200; body = $body } | ConvertTo-Json -Compress
} catch {
  if ($null -ne $_.Exception.Response) {
    $statusCode = [int]$_.Exception.Response.StatusCode
  } else {
    $statusCode = 0
  }
  [pscustomobject]@{
    status_code = $statusCode
    error = $_.Exception.Message
  } | ConvertTo-Json -Compress
  exit 1
}
"""
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        env=environment,
        text=True,
    )
    if completed.returncode != 0:
        try:
            payload_out = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise URLError(completed.stderr or completed.stdout) from exc
        raise URLError(payload_out.get("error", "PowerShell request failed."))
    payload_out = json.loads(completed.stdout)
    body = payload_out.get("body", "{}")
    parsed_body = json.loads(body) if isinstance(body, str) else body
    return int(payload_out["status_code"]), parsed_body


def api_monitor(api_url: str) -> dict[str, Any]:
    """Check that a deployed API can answer health and prediction requests."""
    base_url = api_url.rstrip("/")
    prediction_payload = PredictionRequestExample().as_payload()
    metadata = load_metadata()
    try:
        health_status, health = _request_json(f"{base_url}/health")
        predict_status, prediction = _request_json(
            f"{base_url}/predict",
            method="POST",
            payload=prediction_payload,
        )
        prediction_value = prediction.get("prediction")
        model_version = prediction.get("model_version") or health.get("model_version")
        schema_ok = isinstance(prediction_value, int | float) and bool(model_version)
        response_status = (
            "passed" if health_status == 200 and predict_status == 200 and schema_ok else "failed"
        )
        error = None
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        health_status = None
        predict_status = None
        health = {}
        prediction = {}
        model_version = metadata.get("model_version")
        schema_ok = False
        response_status = "failed"
        error = str(exc)
    retraining_required = response_status != "passed"
    report = {
        "status": "monitored",
        "timestamp_utc": utc_now(),
        "monitoring_mode": "api_aware",
        "production_claim": "api_check_only",
        "api_url": base_url,
        "dataset_name": metadata.get("dataset_name", "UCI Metro Interstate Traffic Volume"),
        "dataset_source": metadata.get("dataset_source", DATA_SOURCE_PAGE),
        "feature_schema": FEATURE_COLUMNS,
        "prediction_request_schema": prediction_payload,
        "model_version": model_version,
        "expected_model_version": metadata.get("model_version"),
        "health": {"status_code": health_status, "response": health},
        "predict": {"status_code": predict_status, "response": prediction},
        "response_status": response_status,
        "prediction_response_schema_valid": schema_ok,
        "retraining_required": retraining_required,
        "retraining_recommended": retraining_required,
        "reason": (
            "Investigate deployed API/model because health or prediction monitoring failed."
            if retraining_required
            else "No retraining investigation required from API health and prediction checks."
        ),
        "error": error,
    }
    write_json(API_MONITORING_REPORT_PATH, report)
    if response_status != "passed":
        raise RuntimeError(f"API-aware monitoring failed: {error or response_status}")
    return report


def parse_args() -> argparse.Namespace:
    """Parse the optional API URL used for Docker or Kind monitoring checks."""
    parser = argparse.ArgumentParser(description="Run offline and optional API-aware monitoring.")
    parser.add_argument("--api-url", default=None, help="Optional deployed API base URL.")
    return parser.parse_args()


def main() -> None:
    """Run offline monitoring and optional API-aware monitoring from the CLI."""
    args = parse_args()
    report = offline_monitor()
    if args.api_url:
        report = {"offline": report, "api": api_monitor(args.api_url)}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
