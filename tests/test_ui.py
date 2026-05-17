from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.main import create_app
from app.schemas import FEATURE_COLUMNS, PredictionRequestExample


class DummyClassifier:
    """Predictable model bundle used to render the UI without loading disk artefacts."""

    classes_ = np.array([0, 1])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([1] * len(frame))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([[0.18, 0.82]] * len(frame))


def create_test_app():
    """Create a Flask app whose UI still uses the real schema and route names."""
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
    # The first page must be usable in the recorded demo without extra setup text:
    # it should show the model, dataset, version, and real endpoint routes.
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
    # This catches a common MLOps/UI failure: adding or renaming a model feature
    # but forgetting to update the browser form used during the demo.
    response = create_test_app().test_client().get("/")
    html = response.get_data(as_text=True)

    for feature_name in FEATURE_COLUMNS:
        assert f'name="{feature_name}"' in html
    assert html.count("data-feature-input") == len(FEATURE_COLUMNS)
    assert "Use Example" in html
    assert "Predict Quality" in html


def test_ui_javascript_handles_unreachable_prediction_api() -> None:
    # If Flask, Docker, or Kind is not reachable, the browser should explain the
    # problem instead of failing silently during the live demo.
    script = Path("app/static/app.js").read_text(encoding="utf-8")

    assert 'endpointUrl("healthUrl", "/health")' in script
    assert 'endpointUrl("predictUrl", "/predict")' in script
    assert "Prediction API is not reachable" in script


def test_use_example_payload_matches_model_schema() -> None:
    # The example button should submit a valid real payload, not a separate mock
    # that bypasses the model's expected feature schema.
    payload = PredictionRequestExample().as_payload()["features"]

    assert set(payload) == set(FEATURE_COLUMNS)
    assert all(isinstance(value, float) for value in payload.values())
