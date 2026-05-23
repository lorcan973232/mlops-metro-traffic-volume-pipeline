"""Compare candidate metrics with recent accepted model versions.

This module supports Continuous Training. It spots harmful drops against previous
accepted versions before a candidate is promoted, while keeping the thresholds
visible in code and reports.
"""

from __future__ import annotations

import json
from typing import Any

from src.versioning import VERSION_MANIFEST_PATH

# ==============================================================================
# Regression checks across accepted versions
# ==============================================================================
#
# Continuous Training should not only check the current model in isolation. These
# helpers compare a candidate against recent accepted versions so a retraining run
# can be rejected if it noticeably weakens the saved model metrics.


def get_recent_versions(count: int = 5) -> list[dict[str, Any]]:
    """Read the most recent accepted model versions from the manifest."""
    if not VERSION_MANIFEST_PATH.exists():
        return []

    manifest = json.loads(VERSION_MANIFEST_PATH.read_text(encoding="utf-8"))
    versions = manifest.get("versions", [])
    return sorted(versions, key=lambda v: v["timestamp"], reverse=True)[:count]


def detect_regressions(
    current_metrics: dict[str, Any], current_version: str
) -> dict[str, Any]:
    """Compare current metrics with recent versions and flag harmful drops."""
    recent_versions = get_recent_versions(5)

    # The current version may already be in the manifest after registration, so
    # exclude it before checking whether the new run is worse than prior metrics.
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

        # Accuracy can move slightly between retraining runs, but a visible drop
        # should be treated as a critical regression for this project.
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

        # Weighted F1 is stricter because it reflects both class labels and is the
        # kind of metric that is easy to compare between model versions.
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

        # Balanced accuracy protects the minority/majority class balance better
        # than plain accuracy, so it is allowed only a small drop.
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

    # Only critical regressions automatically recommend rejection. Other findings
    # are still recorded so they can be reviewed during Continuous Training.
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
    """Print the regression-check thresholds used by Continuous Training."""
    print("Regression detection system initialized")
    print("  Configuration: Max accuracy drop = 0.5%, F1 drop = 0%, Balanced accuracy drop = 2%")


if __name__ == "__main__":
    main()
