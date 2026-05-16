from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cost_benefit_analysis_uses_labelled_simulated_assumptions() -> None:
    subprocess.run([sys.executable, "scripts/cost_benefit_analysis.py"], check=True)

    report_path = Path("reports/business/cost_benefit_report.json")
    summary_path = Path("reports/business/cost_benefit_summary.txt")
    assert report_path.is_file()
    assert summary_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["assumptions"]["assumption_type"] == "SIMULATED_ASSUMPTIONS"
    assert report["computed_from_model"] is True
    assert "confusion_matrix" in report
    for key in ["true_negative", "false_positive", "false_negative", "true_positive"]:
        assert isinstance(report["confusion_matrix"][key], int)
    assert isinstance(report["incremental_value_vs_majority_baseline"], float)
    assert "placeholder" not in json.dumps(report).lower()
