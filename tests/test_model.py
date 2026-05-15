from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import FEATURE_COLUMNS
from src.train import MODEL_HYPERPARAMETERS, RANDOM_STATE, TEST_SIZE, build_pipeline


def sample_training_frame() -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.DataFrame(
        [
            {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.70,
                "citric_acid": 0.00,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "ph": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
            },
            {
                "fixed_acidity": 7.3,
                "volatile_acidity": 0.65,
                "citric_acid": 0.00,
                "residual_sugar": 1.2,
                "chlorides": 0.065,
                "free_sulfur_dioxide": 15.0,
                "total_sulfur_dioxide": 21.0,
                "density": 0.9946,
                "ph": 3.39,
                "sulphates": 0.47,
                "alcohol": 10.0,
            },
            {
                "fixed_acidity": 10.3,
                "volatile_acidity": 0.32,
                "citric_acid": 0.45,
                "residual_sugar": 6.4,
                "chlorides": 0.073,
                "free_sulfur_dioxide": 5.0,
                "total_sulfur_dioxide": 13.0,
                "density": 0.9976,
                "ph": 3.23,
                "sulphates": 0.82,
                "alcohol": 12.6,
            },
            {
                "fixed_acidity": 8.5,
                "volatile_acidity": 0.28,
                "citric_acid": 0.56,
                "residual_sugar": 1.8,
                "chlorides": 0.092,
                "free_sulfur_dioxide": 35.0,
                "total_sulfur_dioxide": 103.0,
                "density": 0.9969,
                "ph": 3.30,
                "sulphates": 0.75,
                "alcohol": 10.5,
            },
        ]
    )
    y = np.array([0, 1, 1, 0])
    return frame[FEATURE_COLUMNS], y


def test_model_pipeline_fits_and_predicts_with_selected_schema() -> None:
    frame, target = sample_training_frame()
    pipeline = build_pipeline()
    pipeline.fit(frame, target)
    predictions = pipeline.predict(frame)
    probabilities = pipeline.predict_proba(frame)
    assert len(predictions) == len(frame)
    assert set(predictions).issubset({0, 1})
    assert probabilities.shape == (len(frame), 2)


def test_model_hyperparameters_are_explicit_and_reproducible() -> None:
    classifier = MODEL_HYPERPARAMETERS["classifier"]
    split = MODEL_HYPERPARAMETERS["train_test_split"]
    preprocessing = MODEL_HYPERPARAMETERS["preprocessing"]

    assert MODEL_HYPERPARAMETERS["algorithm"] == "ExtraTreesClassifier"
    assert classifier["n_estimators"] == 300
    assert classifier["class_weight"] == "balanced"
    assert classifier["random_state"] == RANDOM_STATE
    assert classifier["n_jobs"] == 1
    assert split["test_size"] == TEST_SIZE
    assert split["random_state"] == RANDOM_STATE
    assert split["stratify"] == "quality_label"
    assert preprocessing["numeric_scaler"] == "StandardScaler"
