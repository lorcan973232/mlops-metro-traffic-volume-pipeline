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


def create_test_app():
    return create_app(
        model_bundle={
            "model": DummyModel(),
            "model_version": "test-ui-model-v1",
            "model_path": "models/breast_cancer_classifier.joblib",
            "dataset": {"name": "UCI Breast Cancer Wisconsin Diagnostic"},
            "feature_columns": FEATURE_COLUMNS,
            "class_labels": ("malignant", "benign"),
        }
    )


def test_root_page_renders_professional_model_ui() -> None:
    response = create_test_app().test_client().get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Breast Cancer Diagnostic Classifier" in html
    assert "UCI Breast Cancer Wisconsin Diagnostic" in html
    assert "test-ui-model-v1" in html
    assert "Bike Sharing Demand Predictor" not in html
    assert "season" not in html.lower()
    assert "wind speed" not in html.lower()


def test_root_page_form_fields_match_prediction_schema() -> None:
    response = create_test_app().test_client().get("/")
    html = response.get_data(as_text=True)

    for feature_name in FEATURE_COLUMNS:
        assert f'name="{feature_name}"' in html
    assert html.count("data-feature-input") == len(FEATURE_COLUMNS)
    assert "Use Example" in html
    assert "Predict Diagnosis" in html


def test_use_example_payload_matches_model_schema() -> None:
    payload = PredictionRequestExample().as_payload()["features"]

    assert set(payload) == set(FEATURE_COLUMNS)
    assert all(isinstance(value, float) for value in payload.values())
