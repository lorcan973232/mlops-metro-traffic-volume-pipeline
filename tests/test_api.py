from __future__ import annotations

import numpy as np
import pandas as pd

from app.main import create_app
from app.schemas import FEATURE_COLUMNS, PredictionRequestExample


class DummyModel:
    classes_ = np.array(["malignant", "benign"])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array(["malignant"] * len(frame))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([[0.97, 0.03]] * len(frame))


def test_api_health_and_prediction_use_breast_cancer_schema() -> None:
    app = create_app(
        model_bundle={
            "model": DummyModel(),
            "model_version": "test-model-v1",
            "model_path": "models/breast_cancer_classifier.joblib",
            "dataset": {"name": "UCI Breast Cancer Wisconsin Diagnostic"},
            "feature_columns": FEATURE_COLUMNS,
            "class_labels": ("malignant", "benign"),
        }
    )
    client = app.test_client()
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["feature_count"] == 30
    assert health.get_json()["model_version"] == "test-model-v1"

    prediction = client.post("/predict", json=PredictionRequestExample().as_payload())
    assert prediction.status_code == 200
    assert prediction.get_json()["prediction"] == "malignant"
    assert prediction.get_json()["model_version"] == "test-model-v1"


def test_api_rejects_missing_feature() -> None:
    app = create_app(
        model_bundle={
            "model": DummyModel(),
            "model_version": "test-model-v1",
            "feature_columns": FEATURE_COLUMNS,
            "class_labels": ("malignant", "benign"),
        }
    )
    payload = PredictionRequestExample().as_payload()
    del payload["features"]["mean_radius"]
    response = app.test_client().post("/predict", json=payload)
    assert response.status_code == 400
    assert "Missing feature columns" in response.get_json()["error"]
