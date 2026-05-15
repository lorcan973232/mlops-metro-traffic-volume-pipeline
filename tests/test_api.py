from __future__ import annotations

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


def test_api_health_and_prediction_use_wine_schema() -> None:
    app = create_app(
        model_bundle={
            "model": DummyClassifier(),
            "model_version": "test-wine-model-v1",
            "model_path": "models/wine_quality_classifier.joblib",
            "dataset": {"name": "UCI Wine Quality - Red Wine"},
            "feature_columns": FEATURE_COLUMNS,
            "task_type": "classification",
            "target_labels": {0: "standard quality", 1: "good quality"},
            "target_definition": {"model_target": "quality_label"},
            "classes": [0, 1],
        }
    )
    client = app.test_client()
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["feature_count"] == 11
    assert health.get_json()["task_type"] == "classification"
    assert health.get_json()["model_version"] == "test-wine-model-v1"

    prediction = client.post("/predict", json=PredictionRequestExample().as_payload())
    assert prediction.status_code == 200
    payload = prediction.get_json()
    assert payload["prediction"] == 1
    assert payload["prediction_label"] == "good quality"
    assert payload["confidence"] == 0.82
    assert payload["target"] == "quality_label"
    assert payload["model_version"] == "test-wine-model-v1"


def test_api_rejects_missing_feature() -> None:
    app = create_app(
        model_bundle={
            "model": DummyClassifier(),
            "model_version": "test-wine-model-v1",
            "feature_columns": FEATURE_COLUMNS,
            "task_type": "classification",
            "target_labels": {0: "standard quality", 1: "good quality"},
            "target_definition": {"model_target": "quality_label"},
            "classes": [0, 1],
        }
    )
    payload = PredictionRequestExample().as_payload()
    del payload["features"]["fixed_acidity"]
    response = app.test_client().post("/predict", json=payload)
    assert response.status_code == 400
    assert "Missing feature columns" in response.get_json()["error"]
