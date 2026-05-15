from __future__ import annotations

import numpy as np
import pandas as pd

from app.main import create_app
from app.schemas import FEATURE_COLUMNS, PredictionRequestExample


class DummyRegressor:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([24.5] * len(frame))


def create_test_app():
    return create_app(
        model_bundle={
            "model": DummyRegressor(),
            "model_version": "test-ui-energy-v1",
            "model_path": "models/energy_efficiency_heating_load_regressor.joblib",
            "dataset": {"name": "UCI Energy Efficiency"},
            "feature_columns": FEATURE_COLUMNS,
            "task_type": "regression",
            "target_unit": "heating load",
            "target_definition": {"model_target": "heating_load"},
        }
    )


def test_root_page_renders_clear_energy_ui() -> None:
    response = create_test_app().test_client().get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Building Energy Load Predictor" in html
    assert "UCI Energy Efficiency" in html
    assert "test-ui-energy-v1" in html
    assert "Bike Sharing Demand Predictor" not in html
    assert "Breast Cancer Diagnostic Classifier" not in html


def test_root_page_form_fields_match_prediction_schema() -> None:
    response = create_test_app().test_client().get("/")
    html = response.get_data(as_text=True)

    for feature_name in FEATURE_COLUMNS:
        assert f'name="{feature_name}"' in html
    assert html.count("data-feature-input") == len(FEATURE_COLUMNS)
    assert "Use Example" in html
    assert "Predict Heating Load" in html


def test_use_example_payload_matches_model_schema() -> None:
    payload = PredictionRequestExample().as_payload()["features"]

    assert set(payload) == set(FEATURE_COLUMNS)
    assert all(isinstance(value, float) for value in payload.values())
