from __future__ import annotations

import numpy as np
import pandas as pd

from app.main import create_app
from app.schemas import FEATURE_COLUMNS, PredictionRequestExample


class DummyModel:
    classes_ = np.array(["low", "medium", "high"])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array(["medium"] * len(frame))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([[0.1, 0.8, 0.1]] * len(frame))


def test_api_health_and_prediction_use_wine_quality_schema() -> None:
    app = create_app(
        model_bundle={
            "model": DummyModel(),
            "model_version": "test-model-v1",
            "model_path": "models/wine_quality_classifier.joblib",
            "dataset": {"name": "UCI Wine Quality - white wine"},
            "feature_columns": FEATURE_COLUMNS,
            "class_labels": ("low", "medium", "high"),
        }
    )
    client = app.test_client()
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["feature_count"] == 11
    assert health.get_json()["model_version"] == "test-model-v1"

    prediction = client.post("/predict", json=PredictionRequestExample().as_payload())
    assert prediction.status_code == 200
    assert prediction.get_json()["prediction"] == "medium"
    assert prediction.get_json()["model_version"] == "test-model-v1"


def test_api_rejects_missing_feature() -> None:
    app = create_app(
        model_bundle={
            "model": DummyModel(),
            "model_version": "test-model-v1",
            "feature_columns": FEATURE_COLUMNS,
            "class_labels": ("low", "medium", "high"),
        }
    )
    payload = PredictionRequestExample().as_payload()
    del payload["features"]["alcohol"]
    response = app.test_client().post("/predict", json=payload)
    assert response.status_code == 400
    assert "Missing feature columns" in response.get_json()["error"]
