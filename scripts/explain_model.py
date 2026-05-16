from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, TARGET_LABELS, write_json
from src.preprocess import PROCESSED_DATA_PATH
from src.train import MODEL_PATH, RANDOM_STATE, TEST_SIZE, load_processed_data, train_model

EXPLAINABILITY_DIR = Path("reports/explainability")
SHAP_SUMMARY_PATH = EXPLAINABILITY_DIR / "shap_summary.json"
SHAP_FEATURE_IMPORTANCE_PATH = EXPLAINABILITY_DIR / "shap_feature_importance.json"
LOCAL_EXPLANATION_PATH = EXPLAINABILITY_DIR / "local_explanation_example.json"
FAST_MODE = os.getenv("FAST_MODE", "0") == "1"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _load_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        train_model()
    return joblib.load(MODEL_PATH)


def _test_split() -> tuple[pd.DataFrame, pd.Series]:
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


def _prediction_context(model: Any, row: pd.DataFrame) -> dict[str, Any]:
    prediction = int(model.predict(row)[0])
    probabilities: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        classes = [int(value) for value in model.classes_]
        target_labels = {int(key): value for key, value in TARGET_LABELS.items()}
        proba = model.predict_proba(row)[0]
        probabilities = {
            target_labels.get(class_value, str(class_value)): float(probability)
            for class_value, probability in zip(classes, proba, strict=False)
        }
    return {
        "prediction": prediction,
        "prediction_label": TARGET_LABELS[prediction],
        "probabilities": probabilities,
    }


def _positive_class_shap_values(raw_values: Any, positive_class_index: int) -> np.ndarray:
    if isinstance(raw_values, list):
        return np.asarray(raw_values[positive_class_index])
    values = np.asarray(raw_values)
    if values.ndim == 3:
        return values[:, :, positive_class_index]
    return values


def _expected_value(raw_value: Any, positive_class_index: int) -> float | list[float]:
    if isinstance(raw_value, list):
        return float(raw_value[positive_class_index])
    value = np.asarray(raw_value)
    if value.ndim == 0:
        return float(value)
    if value.size > positive_class_index:
        return float(value[positive_class_index])
    return value.tolist()


def _shap_explanation(bundle: dict[str, Any], x_test: pd.DataFrame) -> dict[str, Any]:
    import shap

    model = bundle["model"]
    classifier = model.named_steps["classifier"]
    transformed = model.named_steps["preprocessor"].transform(x_test)
    sample_size = min(len(x_test), 80 if FAST_MODE else 200)
    sample = transformed[:sample_size]
    explainer = shap.TreeExplainer(classifier)
    raw_values = explainer.shap_values(sample)
    class_index = list(classifier.classes_).index(1)
    shap_values = _positive_class_shap_values(raw_values, class_index)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = {
        feature: float(value)
        for feature, value in sorted(
            zip(FEATURE_COLUMNS, mean_abs, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )
    }
    local_row = x_test.iloc[[0]]
    local_transformed = transformed[:1]
    local_raw_values = explainer.shap_values(local_transformed)
    local_values = _positive_class_shap_values(local_raw_values, class_index)[0]
    local_contributions = {
        feature: float(value)
        for feature, value in sorted(
            zip(FEATURE_COLUMNS, local_values, strict=False),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    }
    return {
        "method": "shap.TreeExplainer",
        "status": "computed",
        "fallback_used": False,
        "sample_size": sample_size,
        "expected_value": _expected_value(explainer.expected_value, class_index),
        "feature_importance": importance,
        "local_contributions": local_contributions,
        "local_input": {key: float(value) for key, value in local_row.iloc[0].to_dict().items()},
    }


def _fallback_explanation(
    bundle: dict[str, Any],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    reason: str,
) -> dict[str, Any]:
    model = bundle["model"]
    repeats = 2 if FAST_MODE else 5
    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="f1_macro",
        n_repeats=repeats,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance = {
        feature: float(value)
        for feature, value in sorted(
            zip(FEATURE_COLUMNS, result.importances_mean, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )
    }
    row = x_test.iloc[[0]].copy()
    baseline_proba = model.predict_proba(row)[0][list(model.classes_).index(1)]
    medians = x_test.median(numeric_only=True)
    local_contributions = {}
    for feature in FEATURE_COLUMNS:
        perturbed = row.copy()
        perturbed.loc[:, feature] = medians[feature]
        perturbed_proba = model.predict_proba(perturbed)[0][list(model.classes_).index(1)]
        local_contributions[feature] = float(baseline_proba - perturbed_proba)
    return {
        "method": "sklearn.permutation_importance",
        "status": "shap_fallback",
        "fallback_used": True,
        "fallback_reason": reason,
        "sample_size": int(len(x_test)),
        "expected_value": None,
        "feature_importance": importance,
        "local_contributions": dict(
            sorted(local_contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        ),
        "local_input": {key: float(value) for key, value in row.iloc[0].to_dict().items()},
    }


def generate_explainability_reports() -> dict[str, Any]:
    bundle = _load_bundle()
    x_test, y_test = _test_split()
    try:
        explanation = _shap_explanation(bundle, x_test)
    except Exception as exc:  # pragma: no cover - fallback is environment dependent.
        explanation = _fallback_explanation(bundle, x_test, y_test, str(exc))
    model = bundle["model"]
    local_row = pd.DataFrame([explanation["local_input"]], columns=FEATURE_COLUMNS)
    prediction = _prediction_context(model, local_row)
    top_features = list(explanation["feature_importance"].items())[:5]
    summary = {
        "status": explanation["status"],
        "generated_at": utc_now(),
        "model_version": bundle.get("model_version", "unknown"),
        "model_path": str(MODEL_PATH),
        "method": explanation["method"],
        "fallback_used": explanation["fallback_used"],
        "fallback_reason": explanation.get("fallback_reason"),
        "sample_size": explanation["sample_size"],
        "target_class_explained": TARGET_LABELS[1],
        "top_features": top_features,
        "interpretation": (
            "Features with larger mean absolute SHAP values have greater average influence "
            "on good-quality predictions for the evaluated sample."
            if not explanation["fallback_used"]
            else "Permutation importance is used because SHAP was unavailable or incompatible."
        ),
        "computed_from_model": True,
    }
    feature_report = {
        "status": explanation["status"],
        "generated_at": summary["generated_at"],
        "method": explanation["method"],
        "feature_importance": explanation["feature_importance"],
        "top_5_features": top_features,
        "computed_from_model": True,
    }
    local_report = {
        "status": explanation["status"],
        "generated_at": summary["generated_at"],
        "method": explanation["method"],
        "expected_value": explanation["expected_value"],
        "input_features": explanation["local_input"],
        "prediction": prediction,
        "feature_contributions": explanation["local_contributions"],
        "top_5_absolute_contributions": list(explanation["local_contributions"].items())[:5],
        "computed_from_model": True,
    }
    write_json(SHAP_SUMMARY_PATH, summary)
    write_json(SHAP_FEATURE_IMPORTANCE_PATH, feature_report)
    write_json(LOCAL_EXPLANATION_PATH, local_report)
    return {
        "summary": summary,
        "feature_importance": feature_report,
        "local_explanation": local_report,
    }


def main() -> None:
    print(json.dumps(generate_explainability_reports(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
