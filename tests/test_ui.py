from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.main import create_app
from app.schemas import FEATURE_COLUMNS, PredictionRequestExample


class DummyClassifier:
    classes_ = np.array([0, 1])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([1] * len(frame))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([[0.18, 0.82]] * len(frame))


def create_test_app():
    return create_app(
        model_bundle={
            "model": DummyClassifier(),
            "model_version": "test-ui-wine-v1",
            "model_path": "models/wine_quality_classifier.joblib",
            "dataset": {"name": "UCI Wine Quality - Red Wine"},
            "feature_columns": FEATURE_COLUMNS,
            "task_type": "classification",
            "target_labels": {0: "standard quality", 1: "good quality"},
            "target_definition": {"model_target": "quality_label"},
            "classes": [0, 1],
        }
    )


def test_root_page_renders_clear_wine_ui() -> None:
    response = create_test_app().test_client().get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Red Wine Quality Classifier" in html
    assert "UCI Wine Quality - Red Wine" in html
    assert "test-ui-wine-v1" in html
    assert "Predict Quality" in html
    assert 'healthUrl: "/health"' in html
    assert 'predictUrl: "/predict"' in html


def test_root_page_form_fields_match_prediction_schema() -> None:
    response = create_test_app().test_client().get("/")
    html = response.get_data(as_text=True)

    for feature_name in FEATURE_COLUMNS:
        assert f'name="{feature_name}"' in html
    assert html.count("data-feature-input") == len(FEATURE_COLUMNS)
    assert "Use Example" in html
    assert "Predict Quality" in html


def test_ui_javascript_handles_unreachable_prediction_api() -> None:
    script = Path("app/static/app.js").read_text(encoding="utf-8")

    assert 'endpointUrl("healthUrl", "/health")' in script
    assert 'endpointUrl("predictUrl", "/predict")' in script
    assert "Prediction API is not reachable" in script


def test_use_example_payload_matches_model_schema() -> None:
    payload = PredictionRequestExample().as_payload()["features"]

    assert set(payload) == set(FEATURE_COLUMNS)
    assert all(isinstance(value, float) for value in payload.values())
