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
from src.data import CLASS_LABELS, DATA_SOURCE_PAGE, FEATURE_COLUMNS, TARGET_COLUMN, write_json
from src.model_registry import MODEL_METADATA_PATH, load_model_metadata
from src.preprocess import CLASS_COLUMN, PROCESSED_DATA_PATH, preprocess_dataset

MONITORING_REPORT_PATH = Path("reports/monitoring/monitoring_report.json")
API_MONITORING_REPORT_PATH = Path("reports/monitoring/api_monitoring_report.json")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def feature_summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        column: {
            "min": float(frame[column].min()),
            "max": float(frame[column].max()),
            "mean": float(frame[column].mean()),
            "std": float(frame[column].std()),
        }
        for column in FEATURE_COLUMNS
    }


def validate_monitoring_schema(frame: pd.DataFrame) -> dict[str, Any]:
    missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    unexpected_columns = [
        column
        for column in frame.columns
        if column not in {*FEATURE_COLUMNS, CLASS_COLUMN, TARGET_COLUMN}
    ]
    non_numeric_columns = [
        column
        for column in FEATURE_COLUMNS
        if column in frame and not pd.api.types.is_numeric_dtype(frame[column])
    ]
    missing_values = int(frame[FEATURE_COLUMNS].isna().sum().sum()) if not missing_columns else None
    checks_passed = (
        not missing_columns
        and not unexpected_columns
        and not non_numeric_columns
        and missing_values == 0
    )
    status = (
        "passed"
        if checks_passed
        else "failed"
    )
    return {
        "status": status,
        "expected_feature_schema": FEATURE_COLUMNS,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "non_numeric_columns": non_numeric_columns,
        "missing_values": missing_values,
    }


def load_metadata() -> dict[str, Any]:
    if MODEL_METADATA_PATH.exists():
        return load_model_metadata()
    return {
        "model_version": "metadata_unavailable",
        "dataset_name": "UCI Breast Cancer Wisconsin Diagnostic",
        "dataset_source": DATA_SOURCE_PAGE,
        "feature_schema": FEATURE_COLUMNS,
        "model_path": "models/breast_cancer_classifier.joblib",
        "metric_summary": {},
        "quality_gate": {"passed": None},
    }


def offline_monitor(processed_path: Path = PROCESSED_DATA_PATH) -> dict[str, Any]:
    if not processed_path.exists():
        preprocess_dataset(output_path=processed_path)
    frame = pd.read_csv(processed_path)
    metadata = load_metadata()
    data_quality = validate_monitoring_schema(frame)
    retraining_required = data_quality["status"] != "passed"
    report = {
        "status": "monitored",
        "timestamp_utc": utc_now(),
        "monitoring_mode": "offline_simulated_monitoring",
        "production_claim": "simulated_only",
        "production_limitation": (
            "No live production telemetry is available in this student artefact; monitoring "
            "uses the selected public dataset schema and deterministic batch checks."
        ),
        "dataset_name": metadata.get("dataset_name", "UCI Breast Cancer Wisconsin Diagnostic"),
        "dataset_source": metadata.get("dataset_source", DATA_SOURCE_PAGE),
        "model_version": metadata.get("model_version"),
        "model_path": metadata.get("model_path"),
        "feature_schema": FEATURE_COLUMNS,
        "prediction_request_schema": {"features": FEATURE_COLUMNS},
        "response_status": "simulated_batch_available",
        "rows": int(len(frame)),
        "data_quality": data_quality,
        "feature_summary": feature_summary(frame),
        "class_distribution": {
            str(key): int(value)
            for key, value in frame[CLASS_COLUMN].value_counts().sort_index().to_dict().items()
        },
        "retraining_required": retraining_required,
        "retraining_recommended": retraining_required,
        "reason": (
            "Investigate retraining because data-quality checks failed."
            if retraining_required
            else "No retraining required from offline schema and data-quality checks."
        ),
    }
    write_json(MONITORING_REPORT_PATH, report)
    return report


def _request_json(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    if os.name == "nt":
        return _request_json_with_powershell(url, method=method, payload=payload)

    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
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
    $response = Invoke-RestMethod `
      -Uri $uri `
      -Method Get `
      -TimeoutSec 10
  }
  [pscustomobject]@{
    status_code = 200
    body = ($response | ConvertTo-Json -Depth 10 -Compress)
  } | ConvertTo-Json -Compress
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
    if isinstance(body, str):
        parsed_body = json.loads(body)
    else:
        parsed_body = body
    return int(payload_out["status_code"]), parsed_body


def api_monitor(api_url: str) -> dict[str, Any]:
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
        schema_ok = prediction_value in set(CLASS_LABELS) and bool(model_version)
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
        "monitoring_mode": "api_aware_monitoring",
        "production_claim": "api_check_only",
        "api_url": base_url,
        "dataset_name": metadata.get("dataset_name", "UCI Breast Cancer Wisconsin Diagnostic"),
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
    parser = argparse.ArgumentParser(description="Run offline and optional API-aware monitoring.")
    parser.add_argument("--api-url", default=None, help="Optional deployed API base URL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = offline_monitor()
    if args.api_url:
        report = {"offline": report, "api": api_monitor(args.api_url)}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
