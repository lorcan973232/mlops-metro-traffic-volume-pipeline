"""Tests for Flask health, prediction, and schema-validation behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.main import create_app
from app.schemas import FEATURE_COLUMNS, PredictionRequestExample


class DummyClassifier:
    """Small predictable model so API tests focus on schema and response contract."""

    classes_ = np.array([0, 1])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([1] * len(frame))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([[0.18, 0.82]] * len(frame))


def test_api_health_and_prediction_use_traffic_schema() -> None:
    # This protects the live demo path: `/health` must prove the model is loaded,
    # and `/predict` must return the same label/confidence fields used by the UI
    # and deployment smoke tests.
    app = create_app(
        model_bundle={
            "model": DummyClassifier(),
            "model_version": "test-traffic-model-v1",
            "model_path": "models/traffic_volume_classifier.joblib",
            "dataset": {"name": "UCI Metro Interstate Traffic Volume"},
            "feature_columns": FEATURE_COLUMNS,
            "task_type": "classification",
            "target_labels": {0: "normal traffic", 1: "high traffic"},
            "target_definition": {"model_target": "high_traffic"},
            "classes": [0, 1],
        }
    )
    client = app.test_client()
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["feature_count"] == len(FEATURE_COLUMNS)
    assert health.get_json()["task_type"] == "classification"
    assert health.get_json()["model_version"] == "test-traffic-model-v1"

    prediction = client.post("/predict", json=PredictionRequestExample().as_payload())
    assert prediction.status_code == 200
    payload = prediction.get_json()
    assert payload["prediction"] == 1
    assert payload["prediction_label"] == "high traffic"
    assert payload["confidence"] == 0.82
    assert payload["target"] == "high_traffic"
    assert payload["model_version"] == "test-traffic-model-v1"


def test_api_rejects_missing_feature() -> None:
    # Missing features should fail before prediction. Otherwise the model could
    # receive shifted or incomplete input and still return a misleading answer.
    app = create_app(
        model_bundle={
            "model": DummyClassifier(),
            "model_version": "test-traffic-model-v1",
            "feature_columns": FEATURE_COLUMNS,
            "task_type": "classification",
            "target_labels": {0: "normal traffic", 1: "high traffic"},
            "target_definition": {"model_target": "high_traffic"},
            "classes": [0, 1],
        }
    )
    payload = PredictionRequestExample().as_payload()
    del payload["features"]["temp"]
    response = app.test_client().post("/predict", json=payload)
    assert response.status_code == 400
    assert "Missing feature columns" in response.get_json()["error"]
