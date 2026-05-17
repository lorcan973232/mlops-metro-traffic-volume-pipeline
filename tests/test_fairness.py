"""Tests for the proxy subgroup audit and its stated limitations."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_fairness_audit_generates_proxy_group_metrics() -> None:
    """Check fairness evidence is computed and avoids protected-attribute claims."""
    subprocess.run([sys.executable, "scripts/fairness_audit.py"], check=True)

    report_path = Path("reports/fairness/fairness_report.json")
    group_path = Path("reports/fairness/group_metrics.json")
    summary_path = Path("reports/fairness/fairness_summary.txt")
    assert report_path.is_file()
    assert group_path.is_file()
    assert summary_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    group_metrics = json.loads(group_path.read_text(encoding="utf-8"))

    assert report["status"] == "completed"
    assert report["dataset_has_protected_attributes"] is False
    assert "not protected characteristics" in report["proxy_group_statement"]
    assert report["computed_from_model"] is True
    assert "alcohol_tertile_proxy" in group_metrics["group_metrics"]
    assert "sulphates_tertile_proxy" in group_metrics["group_metrics"]
    assert "max_equalized_odds_style_gap" in report
    assert isinstance(report["performance_balanced_across_proxy_groups"], bool)
    assert "placeholder" not in json.dumps(report).lower()
