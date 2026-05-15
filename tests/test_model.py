from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import FEATURE_COLUMNS
from src.train import MODEL_HYPERPARAMETERS, RANDOM_STATE, TEST_SIZE, build_pipeline


def sample_training_frame() -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.DataFrame(
        [
            {
                "relative_compactness": 0.76,
                "surface_area": 661.5,
                "wall_area": 416.5,
                "roof_area": 122.5,
                "overall_height": 7.0,
                "orientation": 2,
                "glazing_area": 0.4,
                "glazing_area_distribution": 5,
            },
            {
                "relative_compactness": 0.66,
                "surface_area": 759.5,
                "wall_area": 318.5,
                "roof_area": 220.5,
                "overall_height": 3.5,
                "orientation": 5,
                "glazing_area": 0.1,
                "glazing_area_distribution": 1,
            },
            {
                "relative_compactness": 0.9,
                "surface_area": 563.5,
                "wall_area": 318.5,
                "roof_area": 122.5,
                "overall_height": 7.0,
                "orientation": 3,
                "glazing_area": 0.25,
                "glazing_area_distribution": 3,
            },
            {
                "relative_compactness": 0.62,
                "surface_area": 808.5,
                "wall_area": 367.5,
                "roof_area": 220.5,
                "overall_height": 3.5,
                "orientation": 4,
                "glazing_area": 0.4,
                "glazing_area_distribution": 2,
            },
        ]
    )
    y = np.array([32.0, 11.0, 28.0, 15.0])
    return frame[FEATURE_COLUMNS], y


def test_model_pipeline_fits_and_predicts_with_selected_schema() -> None:
    frame, target = sample_training_frame()
    pipeline = build_pipeline()
    pipeline.fit(frame, target)
    predictions = pipeline.predict(frame)
    assert len(predictions) == len(frame)
    assert np.all(np.isfinite(predictions))


def test_model_hyperparameters_are_explicit_and_reproducible() -> None:
    regressor = MODEL_HYPERPARAMETERS["regressor"]
    split = MODEL_HYPERPARAMETERS["train_test_split"]
    preprocessing = MODEL_HYPERPARAMETERS["preprocessing"]

    assert MODEL_HYPERPARAMETERS["algorithm"] == "GradientBoostingRegressor"
    assert regressor["n_estimators"] == 800
    assert regressor["learning_rate"] == 0.04
    assert regressor["max_depth"] == 4
    assert regressor["random_state"] == RANDOM_STATE
    assert split["test_size"] == TEST_SIZE
    assert split["random_state"] == RANDOM_STATE
    assert preprocessing["categorical_encoder"].startswith("OneHotEncoder")
    assert preprocessing["numeric_scaler"] == "StandardScaler"
