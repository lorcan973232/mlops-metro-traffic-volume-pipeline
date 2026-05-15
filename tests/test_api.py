from __future__ import annotations

import numpy as np
import pandas as pd

from app.main import create_app
from app.schemas import FEATURE_COLUMNS, PredictionRequestExample


class DummyRegressor:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([24.5] * len(frame))


def test_api_health_and_prediction_use_energy_schema() -> None:
    app = create_app(
        model_bundle={
            "model": DummyRegressor(),
            "model_version": "test-energy-model-v1",
            "model_path": "models/energy_efficiency_heating_load_regressor.joblib",
            "dataset": {"name": "UCI Energy Efficiency"},
            "feature_columns": FEATURE_COLUMNS,
            "task_type": "regression",
            "target_unit": "heating load",
            "target_definition": {"model_target": "heating_load"},
        }
    )
    client = app.test_client()
    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["feature_count"] == 8
    assert health.get_json()["task_type"] == "regression"
    assert health.get_json()["model_version"] == "test-energy-model-v1"

    prediction = client.post("/predict", json=PredictionRequestExample().as_payload())
    assert prediction.status_code == 200
    assert prediction.get_json()["prediction"] == 24.5
    assert prediction.get_json()["target"] == "heating_load"
    assert prediction.get_json()["model_version"] == "test-energy-model-v1"


def test_api_rejects_missing_feature() -> None:
    app = create_app(
        model_bundle={
            "model": DummyRegressor(),
            "model_version": "test-energy-model-v1",
            "feature_columns": FEATURE_COLUMNS,
            "task_type": "regression",
            "target_definition": {"model_target": "heating_load"},
        }
    )
    payload = PredictionRequestExample().as_payload()
    del payload["features"]["relative_compactness"]
    response = app.test_client().post("/predict", json=payload)
    assert response.status_code == 400
    assert "Missing feature columns" in response.get_json()["error"]
