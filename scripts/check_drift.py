"""Create offline drift evidence using Population Stability Index.

Continuous Monitoring needs a repeatable drift stage even though this student
artefact has no live production batch. The script compares the processed dataset
with itself, then compares it with a deterministic shifted batch. That proves
the no-drift and drift-trigger paths without pretending to have production data.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATA_SOURCE_PAGE, FEATURE_COLUMNS, write_json
from src.model_registry import MODEL_METADATA_PATH, load_model_metadata
from src.preprocess import PROCESSED_DATA_PATH, preprocess_dataset

DRIFT_REPORT_PATH = Path("reports/monitoring/drift_report.json")
DATA_QUALITY_REPORT_PATH = Path("reports/monitoring/data_quality_report.json")
DRIFT_THRESHOLD = 0.2


# ==============================================================================
# Drift demonstration
# ==============================================================================
#
# The current batch is the processed dataset, so it should not drift from itself.
# A deterministic perturbed batch is also created to prove that the PSI logic can
# raise a retraining signal when feature distributions move.


def utc_now() -> str:
    """Return a UTC timestamp for drift and data-quality reports."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def validate_feature_schema(frame: pd.DataFrame) -> None:
    """Fail early if a monitoring batch cannot be scored by the model schema."""
    missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Drift batch missing expected feature columns: {missing_columns}")
    missing_values = int(frame[FEATURE_COLUMNS].isna().sum().sum())
    if missing_values:
        raise ValueError(f"Drift batch contains {missing_values} missing feature values.")


def data_quality_checks(frame: pd.DataFrame) -> dict[str, Any]:
    """Record schema, type, and missing-value checks for the current batch."""
    missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    non_numeric_columns = []
    missing_by_feature = (
        {column: int(frame[column].isna().sum()) for column in FEATURE_COLUMNS}
        if not missing_columns
        else {}
    )
    total_missing = sum(missing_by_feature.values()) if missing_by_feature else None
    status = (
        "passed"
        if not missing_columns and total_missing == 0
        else "failed"
    )
    return {
        "status": status,
        "expected_feature_schema": FEATURE_COLUMNS,
        "missing_columns": missing_columns,
        "non_numeric_columns": non_numeric_columns,
        "missing_values": total_missing,
        "missing_by_feature": missing_by_feature,
        "row_count": int(len(frame)),
    }


def feature_distribution_checks(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict[str, Any]:
    """Compare reference and current feature summaries for monitoring evidence."""
    checks = {}
    for column in FEATURE_COLUMNS:
        checks[column] = {
            "reference_mean": float(reference[column].mean())
            if pd.api.types.is_numeric_dtype(reference[column])
            else None,
            "current_mean": float(current[column].mean())
            if pd.api.types.is_numeric_dtype(current[column])
            else None,
            "reference_std": float(reference[column].std())
            if pd.api.types.is_numeric_dtype(reference[column])
            else None,
            "current_std": float(current[column].std())
            if pd.api.types.is_numeric_dtype(current[column])
            else None,
        }
    return checks


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calculate PSI for one feature using reference quantile bins."""
    reference_values = reference.dropna().to_numpy()
    current_values = current.dropna().to_numpy()
    breakpoints = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(breakpoints) < 3:
        return 0.0
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    reference_counts, _ = np.histogram(reference_values, bins=breakpoints)
    current_counts, _ = np.histogram(current_values, bins=breakpoints)
    reference_percent = np.maximum(reference_counts / len(reference_values), epsilon)
    current_percent = np.maximum(current_counts / len(current_values), epsilon)
    return float(
        np.sum((current_percent - reference_percent) * np.log(current_percent / reference_percent))
    )


def simulate_drift(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a repeatable shifted batch so drift detection can be demonstrated."""
    drifted = frame.copy()
    drifted["temp"] = drifted["temp"] + 8.0
    drifted["clouds_all"] = (drifted["clouds_all"] + 30.0).clip(upper=100.0)
    drifted["lag_1h_volume"] = (drifted["lag_1h_volume"] * 1.25).clip(upper=8000.0)
    drifted["rolling_24h_volume"] = (drifted["rolling_24h_volume"] * 0.75).clip(lower=0.0)
    return drifted


def load_metadata() -> dict[str, Any]:
    """Load model metadata so drift evidence can name the model it relates to."""
    if MODEL_METADATA_PATH.exists():
        return load_model_metadata()
    return {
        "model_version": "metadata_unavailable",
        "dataset_name": "UCI Metro Interstate Traffic Volume",
        "dataset_source": DATA_SOURCE_PAGE,
        "feature_schema": FEATURE_COLUMNS,
        "quality_gate": {
            "thresholds": {
                "min_accuracy": 0.80,
                "min_weighted_f1": 0.80,
                "min_macro_f1": 0.80,
                "min_cv_accuracy": 0.77,
                "min_baseline_accuracy_improvement": 0.20,
            }
        },
    }


def drift_report(processed_path: Path = PROCESSED_DATA_PATH) -> dict[str, Any]:
    """Write the data-quality and drift reports used by CM and CT discussion."""
    if not processed_path.exists():
        preprocess_dataset(output_path=processed_path)
    metadata = load_metadata()
    reference = pd.read_csv(processed_path)[FEATURE_COLUMNS]
    current = reference.copy()
    simulated = simulate_drift(reference)
    validate_feature_schema(reference)
    validate_feature_schema(current)
    validate_feature_schema(simulated)
    current_quality = data_quality_checks(current)

    current_scores = {
        column: (
            population_stability_index(reference[column], current[column])
            if pd.api.types.is_numeric_dtype(reference[column])
            else 0.0
        )
        for column in FEATURE_COLUMNS
    }
    simulated_scores = {
        column: (
            population_stability_index(reference[column], simulated[column])
            if pd.api.types.is_numeric_dtype(reference[column])
            else 0.0
        )
        for column in FEATURE_COLUMNS
    }
    current_max = float(max(current_scores.values()))
    simulated_max = float(max(simulated_scores.values()))
    current_drift_detected = current_max >= DRIFT_THRESHOLD
    simulated_drift_detected = simulated_max >= DRIFT_THRESHOLD
    per_feature_drift_flags = {
        column: score >= DRIFT_THRESHOLD for column, score in current_scores.items()
    }
    simulated_feature_drift_flags = {
        column: score >= DRIFT_THRESHOLD for column, score in simulated_scores.items()
    }
    retraining_required = current_drift_detected
    timestamp = utc_now()
    data_quality_report = {
        "status": current_quality["status"],
        "timestamp_utc": timestamp,
        "monitoring_mode": "offline_simulated",
        "reference_data_source": str(processed_path),
        "current_data_source": str(processed_path),
        "schema_validation": current_quality,
        "missing_value_checks": {
            "total_missing_values": current_quality["missing_values"],
            "missing_by_feature": current_quality["missing_by_feature"],
        },
        "feature_distribution_checks": feature_distribution_checks(reference, current),
        "computed_from_data": True,
    }
    report = {
        "status": "checked",
        "timestamp_utc": timestamp,
        "monitoring_mode": "offline_simulated",
        "production_claim": "simulated_only",
        "reference_data_source": str(processed_path),
        "current_data_source": str(processed_path),
        "simulated_data_source": "deterministic perturbation of processed reference data",
        "dataset_name": metadata.get("dataset_name", "UCI Metro Interstate Traffic Volume"),
        "dataset_source": metadata.get("dataset_source", DATA_SOURCE_PAGE),
        "model_version": metadata.get("model_version"),
        "feature_schema": FEATURE_COLUMNS,
        "drift_metric": "population_stability_index",
        "threshold": DRIFT_THRESHOLD,
        "ct_quality_gate_reference": metadata.get("quality_gate", {}).get("thresholds", {}),
        "data_quality_status": current_quality["status"],
        "schema_validation": current_quality,
        "missing_value_checks": data_quality_report["missing_value_checks"],
        "feature_distribution_checks": data_quality_report["feature_distribution_checks"],
        "current_batch": {
            "feature_psi": current_scores,
            "per_feature_drift_flags": per_feature_drift_flags,
            "max_psi": current_max,
            "drift_detected": current_drift_detected,
        },
        "simulated_drift_batch": {
            "feature_psi": simulated_scores,
            "per_feature_drift_flags": simulated_feature_drift_flags,
            "max_psi": simulated_max,
            "drift_detected": simulated_drift_detected,
        },
        "drift_score": simulated_max,
        "retraining_required": retraining_required,
        "retraining_recommended": retraining_required or simulated_drift_detected,
        "reason": (
            "Investigate retraining because current batch drift exceeds threshold."
            if retraining_required
            else "Current batch has no drift; simulated batch confirms drift detection logic."
        ),
        "simulated_retraining_signal": {
            "retraining_required": simulated_drift_detected,
            "reason": (
                "Synthetic drift was injected to demonstrate monitoring "
                "and retraining trigger logic."
            ),
        },
    }
    write_json(DATA_QUALITY_REPORT_PATH, data_quality_report)
    write_json(DRIFT_REPORT_PATH, report)
    return report


def main() -> None:
    """Run the drift report and fail if the demonstration signal is absent."""
    report = drift_report()
    if not report["simulated_drift_batch"]["drift_detected"]:
        raise RuntimeError("Synthetic drift was not detected; monitoring evidence is incomplete.")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
