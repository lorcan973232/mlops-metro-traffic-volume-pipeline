"""Create a labelled simulated decision-value report from model errors.

The values in this script are not real winery costs. They are included to show
how a held-out confusion matrix could be discussed in practical terms during a
demo, while keeping the assumptions plainly marked as simulated.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_LABELS, write_json
from src.preprocess import PROCESSED_DATA_PATH
from src.sklearn_compat import load_joblib_bundle
from src.train import MODEL_PATH, RANDOM_STATE, TEST_SIZE, load_processed_data, train_model

BUSINESS_DIR = Path("reports/business")
COST_BENEFIT_REPORT_PATH = BUSINESS_DIR / "cost_benefit_report.json"
COST_BENEFIT_SUMMARY_PATH = BUSINESS_DIR / "cost_benefit_summary.txt"


# ==============================================================================
# Simulated decision-value check
# ==============================================================================
#
# These values are not real business costs. They are included to show how model
# errors from the confusion matrix could be translated into a decision-making
# discussion during the live demo.


def utc_now() -> str:
    """Return a UTC timestamp for the business evidence report."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _load_bundle() -> dict[str, Any]:
    """Load the trained model bundle, training first if the artefact is missing."""
    if not MODEL_PATH.exists():
        train_model()
    return load_joblib_bundle(MODEL_PATH)


def _evaluation_data() -> tuple[Any, Any]:
    """Recreate the held-out data used to count model decisions."""
    data = load_processed_data(PROCESSED_DATA_PATH)
    _, x_test, _, y_test = train_test_split(
        data[FEATURE_COLUMNS],
        data[TARGET_COLUMN],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=data[TARGET_COLUMN],
    )
    return x_test, y_test


def run_cost_benefit_analysis() -> dict[str, Any]:
    """Create a small cost-benefit report from held-out confusion-matrix counts."""
    bundle = _load_bundle()
    model = bundle["model"]
    x_test, y_test = _evaluation_data()
    predictions = model.predict(x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    assumptions = {
        "assumption_type": "SIMULATED_ASSUMPTIONS",
        "unit": "relative decision-value points per held-out wine sample",
        "true_positive_benefit": 5.0,
        "true_negative_benefit": 1.0,
        "false_positive_cost": -4.0,
        "false_negative_cost": -3.0,
        "rationale": (
            "Values are illustrative only. They model a quality-screening scenario where "
            "correctly identifying good wines has higher value, while false premium routing "
            "and missed good wines both carry practical cost."
        ),
    }
    model_value = (
        tp * assumptions["true_positive_benefit"]
        + tn * assumptions["true_negative_benefit"]
        + fp * assumptions["false_positive_cost"]
        + fn * assumptions["false_negative_cost"]
    )
    majority_prediction = int(y_test.mode().iloc[0])
    if majority_prediction == 1:
        baseline_tn, baseline_fp, baseline_fn, baseline_tp = 0, int((y_test == 0).sum()), 0, int(
            (y_test == 1).sum()
        )
    else:
        baseline_tn, baseline_fp, baseline_fn, baseline_tp = int((y_test == 0).sum()), 0, int(
            (y_test == 1).sum()
        ), 0
    baseline_value = (
        baseline_tp * assumptions["true_positive_benefit"]
        + baseline_tn * assumptions["true_negative_benefit"]
        + baseline_fp * assumptions["false_positive_cost"]
        + baseline_fn * assumptions["false_negative_cost"]
    )
    incremental_value = model_value - baseline_value
    generated_at = utc_now()
    report = {
        "status": "completed",
        "generated_at": generated_at,
        "model_version": bundle.get("model_version", "unknown"),
        "model_path": str(MODEL_PATH),
        "use_case": (
            "Demonstration of how a winery or quality-control team could triage samples "
            "for premium review using model predictions."
        ),
        "prediction_action_mapping": {
            TARGET_LABELS[0]: "route_to_standard_quality_review",
            TARGET_LABELS[1]: "route_to_good_quality_or_premium_review",
        },
        "assumptions": assumptions,
        "confusion_matrix": {
            "labels": [TARGET_LABELS[0], TARGET_LABELS[1]],
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "model_decision_value": float(model_value),
        "majority_class_baseline": {
            "predicted_class": TARGET_LABELS[majority_prediction],
            "true_negative": baseline_tn,
            "false_positive": baseline_fp,
            "false_negative": baseline_fn,
            "true_positive": baseline_tp,
            "decision_value": float(baseline_value),
        },
        "incremental_value_vs_majority_baseline": float(incremental_value),
        "practical_value": (
            "Positive simulated value indicates that the model could support prioritised "
            "quality review under the stated assumptions."
            if incremental_value > 0
            else "The model does not beat the majority-class policy under these assumptions."
        ),
        "limitations": [
            "The values are simulated and are not real business figures.",
            (
                "A real deployment would need domain-calibrated costs, intervention "
                "capacity, and risk review."
            ),
            "The analysis uses a held-out public dataset split rather than production outcomes.",
        ],
        "computed_from_model": True,
    }
    summary = "\n".join(
        [
            "Cost-benefit decision analysis",
            "Assumption type: SIMULATED_ASSUMPTIONS",
            f"Model decision value: {model_value:.2f}",
            f"Majority baseline value: {baseline_value:.2f}",
            f"Incremental value: {incremental_value:.2f}",
            report["practical_value"],
        ]
    )
    write_json(COST_BENEFIT_REPORT_PATH, report)
    COST_BENEFIT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    COST_BENEFIT_SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")
    return report


def main() -> None:
    """Run the simulated cost-benefit analysis as a CLI evidence stage."""
    print(json.dumps(run_cost_benefit_analysis(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
