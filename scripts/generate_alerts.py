#!/usr/bin/env python
"""Generate simple alert and incident records from saved reports.

This script is a lightweight Continuous Monitoring support tool. It reads metric
and drift reports already produced by the pipeline, applies the explicit rules
from `src.alerting`, and writes repository-local alert history. It is not wired
to a pager; it exists to show how model evidence could trigger follow-up action.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.alerting import (
    ALERT_RULES,
    append_alert_history,
    check_alert_triggered,
    create_incident,
    generate_alert,
)


def generate_alerts_from_metrics(
    metrics_path: Path = Path("reports/metrics/latest_metrics.json"),
) -> list[dict]:
    """Check latest metrics against alert rules and create incident evidence."""
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        return []

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    alerts = []

    model_version = metrics.get("model_version", "unknown")
    training_timestamp = metrics.get("training_timestamp", "unknown")

    for _rule_name, rule in ALERT_RULES.items():
        if check_alert_triggered(metrics, rule):
            metric_value = metrics.get(rule.metric_name, 0)
            alert = generate_alert(
                rule, metric_value, model_version, training_timestamp
            )
            alerts.append(alert)
            append_alert_history(alert)
            create_incident(alert)

            print(f"[ALERT] [{rule.severity.upper()}]: {rule.name}")
            print(f"   Metric: {rule.metric_name} = {metric_value}")
            print(f"   Threshold: {rule.threshold}")
            print(f"   Action: {alert['recommended_action']}")

    if not alerts:
        print("[OK] No alerts triggered")

    return alerts


def generate_alerts_from_drift(
    drift_path: Path = Path("reports/monitoring/drift_report.json"),
) -> list[dict]:
    """Check the drift report and create drift-related incident evidence."""
    if not drift_path.exists():
        return []

    drift_report = json.loads(drift_path.read_text(encoding="utf-8"))
    alerts = []

    if drift_report.get("drift_detected", False):
        metrics = {
            "max_psi": drift_report.get("max_psi", 0),
            "model_version": drift_report.get("model_version", "unknown"),
        }

        rule = ALERT_RULES["drift_detected"]
        if check_alert_triggered(metrics, rule):
            alert = generate_alert(
                rule,
                drift_report.get("max_psi"),
                metrics["model_version"],
                drift_report.get("timestamp", "unknown"),
            )
            alerts.append(alert)
            append_alert_history(alert)
            create_incident(alert)

            print("[DRIFT ALERT] [medium]: Data drift detected")
            print(f"   Max PSI: {drift_report.get('max_psi')}")

    return alerts


def main() -> None:
    """Run metric and drift alert generation from the command line."""
    print("Generating alerts from metrics...")
    alerts_from_metrics = generate_alerts_from_metrics()

    print("\nGenerating alerts from drift detection...")
    alerts_from_drift = generate_alerts_from_drift()

    total_alerts = len(alerts_from_metrics) + len(alerts_from_drift)
    print(f"\nTotal alerts generated: {total_alerts}")


if __name__ == "__main__":
    main()
