from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, write_json
from src.preprocess import PROCESSED_DATA_PATH
from src.train import MODEL_PATH, RANDOM_STATE, TEST_SIZE, load_processed_data, train_model

FAIRNESS_DIR = Path("reports/fairness")
FAIRNESS_REPORT_PATH = FAIRNESS_DIR / "fairness_report.json"
GROUP_METRICS_PATH = FAIRNESS_DIR / "group_metrics.json"
FAIRNESS_SUMMARY_PATH = FAIRNESS_DIR / "fairness_summary.txt"


# ==============================================================================
# Proxy subgroup audit
# ==============================================================================
#
# The wine dataset has no demographic protected attributes. This audit therefore
# uses operational proxy groups, such as alcohol and sulphates tertiles, to show
# how subgroup checks would be wired into the pipeline without making legal or
# demographic fairness claims.


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _load_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        train_model()
    return joblib.load(MODEL_PATH)


def _test_frame() -> tuple[pd.DataFrame, pd.Series]:
    data = load_processed_data(PROCESSED_DATA_PATH)
    _, x_test, _, y_test = train_test_split(
        data[FEATURE_COLUMNS],
        data[TARGET_COLUMN],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=data[TARGET_COLUMN],
    )
    return x_test.copy(), y_test.reset_index(drop=True)


def _bin_feature(series: pd.Series, labels: list[str]) -> pd.Series:
    return pd.qcut(series, q=len(labels), labels=labels, duplicates="drop").astype(str)


def _binary_rates(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "true_positive_rate": float(tpr),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
    }


def _group_metric_rows(frame: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    """Calculate model performance separately for the proxy feature groups."""
    grouped_reports: dict[str, Any] = {}
    grouping_columns = {
        "alcohol_tertile_proxy": _bin_feature(frame["alcohol"], ["low", "medium", "high"]),
        "sulphates_tertile_proxy": _bin_feature(frame["sulphates"], ["low", "medium", "high"]),
    }
    for grouping_name, groups in grouping_columns.items():
        rows = {}
        for group_name in sorted(groups.unique()):
            mask = groups == group_name
            group_y = y_true[mask.to_numpy()]
            group_pred = y_pred[mask.to_numpy()]
            rates = _binary_rates(group_y, group_pred)
            rows[group_name] = {
                "count": int(mask.sum()),
                "accuracy": float(accuracy_score(group_y, group_pred)),
                "precision": float(precision_score(group_y, group_pred, zero_division=0)),
                "recall": float(recall_score(group_y, group_pred, zero_division=0)),
                "f1": float(f1_score(group_y, group_pred, zero_division=0)),
                **rates,
            }
        grouped_reports[grouping_name] = rows
    return grouped_reports


def _disparities(group_metrics: dict[str, Any]) -> dict[str, Any]:
    report = {}
    for grouping_name, groups in group_metrics.items():
        metric_gaps = {}
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "true_positive_rate",
            "false_positive_rate",
            "false_negative_rate",
        ]:
            values = [group[metric] for group in groups.values()]
            metric_gaps[f"{metric}_gap"] = float(max(values) - min(values))
        report[grouping_name] = {
            **metric_gaps,
            "equalized_odds_style_gap": float(
                max(metric_gaps["true_positive_rate_gap"], metric_gaps["false_positive_rate_gap"])
            ),
            "largest_gap_metric": max(metric_gaps.items(), key=lambda item: item[1])[0],
        }
    return report


def run_fairness_audit() -> dict[str, Any]:
    """Write the proxy subgroup audit reports used in the marker evidence pack."""
    bundle = _load_bundle()
    x_test, y_test = _test_frame()
    model = bundle["model"]
    predictions = pd.Series(model.predict(x_test))
    group_metrics = _group_metric_rows(x_test, y_test, predictions)
    disparities = _disparities(group_metrics)
    max_gap = max(
        group["equalized_odds_style_gap"]
        for group in disparities.values()
    )
    balanced = max_gap < 0.15
    generated_at = utc_now()
    report = {
        "status": "completed",
        "generated_at": generated_at,
        "model_version": bundle.get("model_version", "unknown"),
        "model_path": str(MODEL_PATH),
        "dataset_has_protected_attributes": False,
        "protected_attribute_statement": (
            "The UCI red wine quality dataset contains physicochemical measurements and a "
            "quality score; it does not contain demographic protected attributes."
        ),
        "proxy_group_statement": (
            "This is a proxy subgroup performance audit using non-sensitive operational "
            "feature bins. These proxy groups are not protected characteristics."
        ),
        "grouping_variables": ["alcohol_tertile_proxy", "sulphates_tertile_proxy"],
        "group_metrics_path": str(GROUP_METRICS_PATH),
        "disparities": disparities,
        "max_equalized_odds_style_gap": float(max_gap),
        "performance_balanced_across_proxy_groups": balanced,
        "interpretation": (
            "Proxy subgroup performance is reasonably balanced under the 0.15 gap threshold."
            if balanced
            else "At least one proxy subgroup gap exceeds 0.15; review subgroup errors before use."
        ),
        "limitations": [
            "No demographic protected attributes are available in this dataset.",
            "Proxy feature bins cannot support legal or demographic fairness claims.",
            "Small subgroup counts can make precision, recall, and F1 volatile.",
        ],
        "computed_from_model": True,
    }
    group_report = {
        "status": "completed",
        "generated_at": generated_at,
        "group_metrics": group_metrics,
        "disparities": disparities,
        "computed_from_model": True,
    }
    summary_text = "\n".join(
        [
            "Fairness proxy subgroup audit",
            f"Model version: {report['model_version']}",
            "Protected attributes present: no",
            "Proxy groups: alcohol tertiles and sulphates tertiles",
            f"Maximum equalized-odds-style gap: {max_gap:.4f}",
            f"Balanced under 0.15 threshold: {balanced}",
            report["interpretation"],
        ]
    )
    write_json(FAIRNESS_REPORT_PATH, report)
    write_json(GROUP_METRICS_PATH, group_report)
    FAIRNESS_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAIRNESS_SUMMARY_PATH.write_text(summary_text + "\n", encoding="utf-8")
    return {"fairness_report": report, "group_metrics": group_report}


def main() -> None:
    print(json.dumps(run_fairness_audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
