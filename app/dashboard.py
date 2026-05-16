from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, jsonify, render_template

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/dashboard")


def get_version_metrics() -> list[dict]:
    """Gather metrics from all versions for time-series visualization."""
    from src.versioning import VERSION_MANIFEST_PATH

    if not VERSION_MANIFEST_PATH.exists():
        return []

    manifest = json.loads(VERSION_MANIFEST_PATH.read_text(encoding="utf-8"))
    versions = sorted(manifest.get("versions", []), key=lambda v: v["timestamp"])

    return [
        {
            "version": v["version"],
            "timestamp": v["timestamp"],
            "accuracy": v["metrics"].get("accuracy"),
            "balanced_accuracy": v["metrics"].get("balanced_accuracy"),
            "f1_macro": v["metrics"].get("f1_macro"),
            "f1_weighted": v["metrics"].get("f1_weighted"),
            "roc_auc": v["metrics"].get("roc_auc"),
            "status": v["status"],
        }
        for v in versions
    ]


def get_latest_metrics() -> dict:
    """Get latest evaluation metrics."""
    metrics_path = Path("reports/metrics/latest_metrics.json")
    if not metrics_path.exists():
        return {}

    return json.loads(metrics_path.read_text(encoding="utf-8"))


def get_sla_metrics() -> dict:
    """Get latest SLA/benchmark metrics."""
    sla_path = Path("reports/benchmarks/api_sla_report.json")
    if not sla_path.exists():
        return {}

    return json.loads(sla_path.read_text(encoding="utf-8"))


def get_drift_metrics() -> dict:
    """Get latest drift detection results."""
    drift_path = Path("reports/monitoring/drift_report.json")
    if not drift_path.exists():
        return {}

    return json.loads(drift_path.read_text(encoding="utf-8"))


def get_fairness_metrics() -> dict:
    """Get fairness analysis results."""
    fairness_path = Path("reports/metrics/fairness_analysis.json")
    if not fairness_path.exists():
        return {}

    return json.loads(fairness_path.read_text(encoding="utf-8"))


@dashboard_bp.route("/")
def index() -> str:
    """Render main dashboard page."""
    versions = get_version_metrics()
    latest_metrics = get_latest_metrics()
    sla_metrics = get_sla_metrics()
    drift_metrics = get_drift_metrics()
    fairness_metrics = get_fairness_metrics()

    return render_template(
        "dashboard.html",
        versions=versions,
        latest_metrics=latest_metrics,
        sla_metrics=sla_metrics,
        drift_metrics=drift_metrics,
        fairness_metrics=fairness_metrics,
    )


@dashboard_bp.route("/api/metrics")
def api_metrics() -> dict:
    """Return JSON metrics for charts."""
    versions = get_version_metrics()
    return jsonify(
        {
            "versions": versions,
            "latest": get_latest_metrics(),
            "sla": get_sla_metrics(),
            "drift": get_drift_metrics(),
            "fairness": get_fairness_metrics(),
        }
    )


@dashboard_bp.route("/api/version/<version>")
def api_version_detail(version: str) -> dict:
    """Return detailed metrics for a specific version."""
    from src.versioning import get_version_record

    record = get_version_record(version)
    if not record:
        return jsonify({"error": "Version not found"}), 404

    return jsonify(record)


def init_app(app) -> None:
    """Register dashboard blueprint with Flask app."""
    app.register_blueprint(dashboard_bp)
