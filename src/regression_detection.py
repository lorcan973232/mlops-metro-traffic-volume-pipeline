from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.versioning import get_version_record, VERSION_MANIFEST_PATH


def get_recent_versions(count: int = 5) -> list[dict[str, Any]]:
    """Get list of last N promoted versions for comparison."""
    if not VERSION_MANIFEST_PATH.exists():
        return []

    manifest = json.loads(VERSION_MANIFEST_PATH.read_text(encoding="utf-8"))
    versions = manifest.get("versions", [])
    return sorted(versions, key=lambda v: v["timestamp"], reverse=True)[:count]


def detect_regressions(
    current_metrics: dict[str, Any], current_version: str
) -> dict[str, Any]:
    """Detect performance regressions comparing to recent versions."""
    recent_versions = get_recent_versions(5)

    # Filter out current version from comparison
    comparison_versions = [v for v in recent_versions if v["version"] != current_version]

    if not comparison_versions:
        return {
            "current_version": current_version,
            "comparison_versions": [],
            "regressions": [],
            "overall_status": "no_prior_versions",
            "recommend_rejection": False,
        }

    regressions = []

    for prior_version_record in comparison_versions:
        prior_metrics = prior_version_record.get("metrics", {})

        # Check accuracy (can tolerate max 0.5% drop)
        accuracy_delta = current_metrics.get("accuracy", 0) - prior_metrics.get(
            "accuracy", 0
        )
        if accuracy_delta < -0.005:  # More than 0.5% drop
            regressions.append(
                {
                    "metric": "accuracy",
                    "previous_value": prior_metrics.get("accuracy"),
                    "current_value": current_metrics.get("accuracy"),
                    "delta": accuracy_delta,
                    "acceptable": False,
                    "severity": "critical",
                    "comparison_version": prior_version_record["version"],
                }
            )

        # Check F1 (cannot drop at all)
        f1_delta = current_metrics.get("f1_weighted", 0) - prior_metrics.get(
            "f1_weighted", 0
        )
        if f1_delta < 0:  # Any F1 drop
            regressions.append(
                {
                    "metric": "f1_weighted",
                    "previous_value": prior_metrics.get("f1_weighted"),
                    "current_value": current_metrics.get("f1_weighted"),
                    "delta": f1_delta,
                    "acceptable": False,
                    "severity": "high",
                    "comparison_version": prior_version_record["version"],
                }
            )

        # Check balanced accuracy (cannot drop more than 2%)
        bal_acc_delta = current_metrics.get("balanced_accuracy", 0) - prior_metrics.get(
            "balanced_accuracy", 0
        )
        if bal_acc_delta < -0.02:  # More than 2% drop
            regressions.append(
                {
                    "metric": "balanced_accuracy",
                    "previous_value": prior_metrics.get("balanced_accuracy"),
                    "current_value": current_metrics.get("balanced_accuracy"),
                    "delta": bal_acc_delta,
                    "acceptable": False,
                    "severity": "high",
                    "comparison_version": prior_version_record["version"],
                }
            )

    # Determine overall status
    critical_regressions = [r for r in regressions if r["severity"] == "critical"]
    overall_status = (
        "critical_regressions" if critical_regressions else "acceptable"
    )
    recommend_rejection = len(critical_regressions) > 0

    return {
        "current_version": current_version,
        "comparison_versions": [v["version"] for v in comparison_versions],
        "regressions": regressions,
        "overall_status": overall_status,
        "recommend_rejection": recommend_rejection,
    }


def main() -> None:
    print("Regression detection system initialized")
    print(f"  Configuration: Max accuracy drop = 0.5%, F1 drop = 0%, Balanced accuracy drop = 2%")


if __name__ == "__main__":
    main()
