"""Tests for the browser demo surface and its API contract."""

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
        """Return deterministic labels for browser rendering tests."""
        return np.array([1] * len(frame))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return fixed probabilities in the same shape as the trained model."""
        return np.array([[0.18, 0.82]] * len(frame))


def create_test_app():
    """Create a Flask app whose UI still uses the real schema and route names."""
    return create_app(
        model_bundle={
            "model": DummyClassifier(),
            "model_version": "test-ui-traffic-v1",
            "model_path": "models/traffic_volume_classifier.joblib",
            "dataset": {"name": "UCI Metro Interstate Traffic Volume"},
            "feature_columns": FEATURE_COLUMNS,
            "task_type": "classification",
            "target_labels": {0: "normal traffic", 1: "high traffic"},
            "target_definition": {"model_target": "high_traffic"},
            "classes": [0, 1],
        }
    )


def test_root_page_renders_clear_traffic_ui() -> None:
    """Check the home page shows model, dataset, and route details."""
    # The first page must be usable in the recorded demo without extra setup text:
    # it should show the model, dataset, version, and real endpoint routes.
    response = create_test_app().test_client().get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Traffic Volume Predictor" in html
    assert "UCI Metro Interstate Traffic Volume" in html
    assert "test-ui-traffic-v1" in html
    assert "Predict Traffic" in html
    assert 'healthUrl: "/health"' in html
    assert 'predictUrl: "/predict"' in html


def test_root_page_form_fields_match_prediction_schema() -> None:
    """Check the browser form fields match the model feature schema."""
    # This catches a common MLOps/UI failure: adding or renaming a model feature
    # but forgetting to update the browser form used during the demo.
    response = create_test_app().test_client().get("/")
    html = response.get_data(as_text=True)

    for feature_name in FEATURE_COLUMNS:
        assert f'name="{feature_name}"' in html
    assert html.count("data-feature-input") == len(FEATURE_COLUMNS)
    assert "Use Example" in html
    assert "Predict Traffic" in html


def test_dashboard_page_renders_current_report_schema() -> None:
    """Check the optional dashboard renders the saved report files."""
    response = create_test_app().test_client().get("/dashboard/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "MLOps Model Performance Dashboard" in html
    assert "Class-Balance Metrics" in html
    assert "Data Drift Status" in html
    assert "Normal Traffic" in html
    assert "High Traffic" in html


def test_ui_javascript_handles_unreachable_prediction_api() -> None:
    """Check the browser explains API connection failures clearly."""
    # If Flask, Docker, or Kind is not reachable, the browser should explain the
    # problem instead of failing silently during the demo.
    script = Path("app/static/app.js").read_text(encoding="utf-8")

    assert 'endpointUrl("healthUrl", "/health")' in script
    assert 'endpointUrl("predictUrl", "/predict")' in script
    assert "Prediction API is not reachable" in script


def test_use_example_payload_matches_model_schema() -> None:
    """Check the example button payload stays aligned with `/predict`."""
    # The example button should submit a valid real payload, not a separate mock
    # that bypasses the model's expected feature schema.
    payload = PredictionRequestExample().as_payload()["features"]

    assert set(payload) == set(FEATURE_COLUMNS)
    assert isinstance(payload["weather_main"], str)
    assert all(isinstance(value, float) for key, value in payload.items() if key != "weather_main")
