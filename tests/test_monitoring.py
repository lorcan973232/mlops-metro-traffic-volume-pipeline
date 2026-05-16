from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_monitoring_and_drift_scripts_generate_required_reports() -> None:
    subprocess.run([sys.executable, "scripts/monitor.py"], check=True)
    subprocess.run([sys.executable, "scripts/check_drift.py"], check=True)

    monitoring_path = Path("reports/monitoring/monitoring_report.json")
    drift_path = Path("reports/monitoring/drift_report.json")
    data_quality_path = Path("reports/monitoring/data_quality_report.json")
    assert monitoring_path.is_file()
    assert drift_path.is_file()
    assert data_quality_path.is_file()

    monitoring = json.loads(monitoring_path.read_text(encoding="utf-8"))
    drift = json.loads(drift_path.read_text(encoding="utf-8"))
    data_quality = json.loads(data_quality_path.read_text(encoding="utf-8"))

    assert monitoring["monitoring_mode"] == "offline_simulated"
    assert monitoring["retraining_required"] is False
    assert data_quality["schema_validation"]["status"] == "passed"
    assert data_quality["missing_value_checks"]["total_missing_values"] == 0
    assert drift["drift_metric"] == "population_stability_index"
    assert drift["current_batch"]["drift_detected"] is False
    assert drift["simulated_drift_batch"]["drift_detected"] is True
    assert isinstance(drift["retraining_required"], bool)
    assert isinstance(drift["retraining_recommended"], bool)
