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
DRIFT_THRESHOLD = 0.2


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def validate_feature_schema(frame: pd.DataFrame) -> None:
    missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Drift batch missing expected feature columns: {missing_columns}")
    non_numeric_columns = [
        column for column in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric_columns:
        raise ValueError(f"Drift batch contains non-numeric feature columns: {non_numeric_columns}")
    missing_values = int(frame[FEATURE_COLUMNS].isna().sum().sum())
    if missing_values:
        raise ValueError(f"Drift batch contains {missing_values} missing feature values.")


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
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
    drifted = frame.copy()
    drifted["volatile_acidity"] = (drifted["volatile_acidity"] * 1.55).clip(upper=1.6)
    drifted["chlorides"] = (drifted["chlorides"] * 1.7).clip(upper=0.7)
    drifted["sulphates"] = (drifted["sulphates"] * 1.45).clip(upper=2.2)
    drifted["alcohol"] = (drifted["alcohol"] + 1.8).clip(upper=16.0)
    return drifted


def load_metadata() -> dict[str, Any]:
    if MODEL_METADATA_PATH.exists():
        return load_model_metadata()
    return {
        "model_version": "metadata_unavailable",
        "dataset_name": "UCI Wine Quality - Red Wine",
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
    if not processed_path.exists():
        preprocess_dataset(output_path=processed_path)
    metadata = load_metadata()
    reference = pd.read_csv(processed_path)[FEATURE_COLUMNS]
    current = reference.copy()
    simulated = simulate_drift(reference)
    validate_feature_schema(reference)
    validate_feature_schema(current)
    validate_feature_schema(simulated)

    current_scores = {
        column: population_stability_index(reference[column], current[column])
        for column in FEATURE_COLUMNS
    }
    simulated_scores = {
        column: population_stability_index(reference[column], simulated[column])
        for column in FEATURE_COLUMNS
    }
    current_max = float(max(current_scores.values()))
    simulated_max = float(max(simulated_scores.values()))
    current_drift_detected = current_max >= DRIFT_THRESHOLD
    simulated_drift_detected = simulated_max >= DRIFT_THRESHOLD
    retraining_required = current_drift_detected
    report = {
        "status": "checked",
        "timestamp_utc": utc_now(),
        "monitoring_mode": "simulated_drift_check",
        "production_claim": "simulated_only",
        "dataset_name": metadata.get("dataset_name", "UCI Wine Quality - Red Wine"),
        "dataset_source": metadata.get("dataset_source", DATA_SOURCE_PAGE),
        "model_version": metadata.get("model_version"),
        "feature_schema": FEATURE_COLUMNS,
        "drift_metric": "population_stability_index",
        "threshold": DRIFT_THRESHOLD,
        "ct_quality_gate_reference": metadata.get("quality_gate", {}).get("thresholds", {}),
        "data_quality_status": "passed",
        "current_batch": {
            "feature_psi": current_scores,
            "max_psi": current_max,
            "drift_detected": current_drift_detected,
        },
        "simulated_drift_batch": {
            "feature_psi": simulated_scores,
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
    write_json(DRIFT_REPORT_PATH, report)
    return report


def main() -> None:
    report = drift_report()
    if not report["simulated_drift_batch"]["drift_detected"]:
        raise RuntimeError("Synthetic drift was not detected; monitoring evidence is incomplete.")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
