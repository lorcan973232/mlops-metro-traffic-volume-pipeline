"""Turn saved metric and monitoring checks into simple incident records.

The alerting layer is deliberately lightweight. It is not a real pager or
production incident system; it gives the project a simple example of how SLA,
drift, fairness-proxy, and quality-gate failures could trigger follow-up records
under `reports/alerts/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALERTS_PATH = Path("reports/alerts")
ALERT_HISTORY_PATH = ALERTS_PATH / "alert_history.json"
INCIDENTS_PATH = ALERTS_PATH / "incidents.json"


# ==============================================================================
# Lightweight alert records
# ==============================================================================
#
# These rules turn monitoring, drift, fairness, and metric checks into simple
# incident evidence. They are not connected to a real pager; they show how a
# Continuous Monitoring stage could flag conditions that would matter before a
# model is promoted or shown in the demo.


@dataclass
class AlertRule:
    """Define one threshold check that can create a readable alert."""

    name: str
    metric_name: str
    threshold: float
    operator: str  # "greater_than", "less_than", "equals"
    severity: str  # "low", "medium", "high", "critical"
    description: str


# The thresholds are intentionally explicit so the report shows which model or
# service condition would trigger an incident record.
ALERT_RULES: dict[str, AlertRule] = {
    "sla_breach": AlertRule(
        name="sla_breach",
        metric_name="p99_latency_ms",
        threshold=200.0,
        operator="greater_than",
        severity="high",
        description="API p99 latency exceeds 200ms SLA threshold",
    ),
    "drift_detected": AlertRule(
        name="drift_detected",
        metric_name="max_psi",
        threshold=0.2,
        operator="greater_than",
        severity="medium",
        description="Data drift detected: PSI exceeds 0.2 threshold",
    ),
    "fairness_degradation": AlertRule(
        name="fairness_degradation",
        metric_name="f1_disparity",
        threshold=0.10,
        operator="greater_than",
        severity="medium",
        description="Fairness degradation: F1 disparity between classes exceeds 10%",
    ),
    "accuracy_drop": AlertRule(
        name="accuracy_drop",
        metric_name="accuracy",
        threshold=0.80,
        operator="less_than",
        severity="critical",
        description="Model accuracy drops below 80% threshold",
    ),
    "balanced_accuracy_drop": AlertRule(
        name="balanced_accuracy_drop",
        metric_name="balanced_accuracy",
        threshold=0.80,
        operator="less_than",
        severity="high",
        description="Balanced accuracy drops below 80% threshold",
    ),
}


def _evaluate_condition(value: float, threshold: float, operator: str) -> bool:
    """Evaluate a metric against the rule operator without hidden logic."""
    if operator == "greater_than":
        return value > threshold
    elif operator == "less_than":
        return value < threshold
    elif operator == "equals":
        return value == threshold
    else:
        raise ValueError(f"Unknown operator: {operator}")


def check_alert_triggered(metrics: dict[str, Any], rule: AlertRule) -> bool:
    """Check whether a report metric is present and crosses a rule threshold."""
    if rule.metric_name not in metrics:
        return False

    value = metrics[rule.metric_name]
    return _evaluate_condition(value, rule.threshold, rule.operator)


def generate_alert(
    rule: AlertRule, metric_value: float, model_version: str, training_timestamp: str
) -> dict[str, Any]:
    """Create the JSON alert record saved by the monitoring evidence path."""
    import uuid

    return {
        "alert_id": str(uuid.uuid4())[:12],
        "timestamp": training_timestamp,
        "severity": rule.severity,
        "rule_name": rule.name,
        "metric_name": rule.metric_name,
        "metric_value": metric_value,
        "threshold": rule.threshold,
        "triggered": True,
        "model_version": model_version,
        "description": rule.description,
        "status": "open",
        "recommended_action": _get_recommended_action(rule),
    }


def _get_recommended_action(rule: AlertRule) -> str:
    """Explain what the student should inspect if the alert triggers."""
    actions = {
        "sla_breach": "Investigate API performance; profile prediction latency",
        "drift_detected": "Trigger immediate model retraining with latest data",
        "fairness_degradation": "Review model for bias; retrain if fairness cannot be improved",
        "accuracy_drop": "BLOCK model promotion; investigate model regression",
        "balanced_accuracy_drop": "Review cross-class performance; may indicate fairness issues",
    }
    return actions.get(rule.name, "Investigate alert condition")


def append_alert_history(alert: dict[str, Any]) -> None:
    """Append an alert to the repository-local history file under reports."""
    ALERTS_PATH.mkdir(parents=True, exist_ok=True)

    history = []
    if ALERT_HISTORY_PATH.exists():
        history = json.loads(ALERT_HISTORY_PATH.read_text(encoding="utf-8"))

    history.append(alert)
    ALERT_HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def get_open_incidents() -> list[dict[str, Any]]:
    """Return unresolved incidents for dashboard or demo inspection."""
    if not INCIDENTS_PATH.exists():
        return []

    incidents = json.loads(INCIDENTS_PATH.read_text(encoding="utf-8"))
    return [inc for inc in incidents if inc.get("status") == "open"]


def create_incident(alert: dict[str, Any]) -> None:
    """Convert an alert into an incident record for model-management evidence."""
    import uuid

    ALERTS_PATH.mkdir(parents=True, exist_ok=True)

    incidents = []
    if INCIDENTS_PATH.exists():
        incidents = json.loads(INCIDENTS_PATH.read_text(encoding="utf-8"))

    incident = {
        "incident_id": f"INC-{uuid.uuid4().hex[:8].upper()}",
        "created_at": alert["timestamp"],
        "alert_id": alert["alert_id"],
        "alert_rule": alert["rule_name"],
        "severity": alert["severity"],
        "model_version": alert["model_version"],
        "metrics": {
            alert["metric_name"]: alert["metric_value"],
            "threshold": alert["threshold"],
        },
        "status": "open",
        "action_taken": alert["recommended_action"],
        "resolved_at": None,
    }

    incidents.append(incident)
    INCIDENTS_PATH.write_text(json.dumps(incidents, indent=2), encoding="utf-8")


def main() -> None:
    """Print the alert rules so the monitoring design can be inspected quickly."""
    print("Available alert rules:")
    for rule_name, rule in ALERT_RULES.items():
        print(f"  - {rule_name}: {rule.description}")


if __name__ == "__main__":
    main()
