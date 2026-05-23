"""Tests for the traffic training pipeline contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.schemas import PredictionRequestExample
from src.data import FEATURE_COLUMNS
from src.train import (
    MODEL_HYPERPARAMETERS,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
    build_pipeline,
)


def sample_training_frame() -> tuple[pd.DataFrame, np.ndarray]:
    """Create a small traffic-like frame for fast pipeline contract tests."""
    base = PredictionRequestExample().__dict__.copy()
    rows = []
    labels = []
    for i in range(30):
        row = base.copy()
        row["hour"] = float(i % 24)
        row["month"] = float((i % 12) + 1)
        row["day_of_week"] = float(i % 7)
        row["weather_main"] = "Clear" if i % 2 else "Clouds"
        row["lag_1h_volume"] = float(1800 + i * 160)
        row["lag_24h_volume"] = float(1700 + i * 150)
        row["lag_168h_volume"] = float(1600 + i * 140)
        row["rolling_3h_volume"] = float(1750 + i * 150)
        row["rolling_24h_volume"] = float(1900 + i * 120)
        rows.append(row)
        labels.append(1 if row["lag_1h_volume"] >= 3800 else 0)
    return pd.DataFrame(rows)[FEATURE_COLUMNS], np.array(labels)


def test_model_pipeline_fits_and_predicts_with_selected_schema() -> None:
    """Check the model pipeline fits and predicts with the saved feature order."""
    frame, target = sample_training_frame()
    pipeline = build_pipeline()
    pipeline.fit(frame, target)
    predictions = pipeline.predict(frame)
    probabilities = pipeline.predict_proba(frame)
    assert len(predictions) == len(frame)
    assert set(predictions).issubset({0, 1})
    assert probabilities.shape == (len(frame), 2)


def test_model_hyperparameters_are_explicit_and_reproducible() -> None:
    """Check key model settings are fixed for repeatable training reports."""
    classifier = MODEL_HYPERPARAMETERS["classifier"]
    split = MODEL_HYPERPARAMETERS["train_test_split"]
    preprocessing = MODEL_HYPERPARAMETERS["preprocessing"]

    assert MODEL_HYPERPARAMETERS["algorithm"] == "HistGradientBoostingClassifier"
    assert classifier["max_iter"] == 320
    assert classifier["class_weight"] == "balanced"
    assert classifier["random_state"] == RANDOM_STATE
    assert split["test_size"] == TEST_SIZE
    assert split["validation_size"] == VALIDATION_SIZE
    assert split["random_state"] == RANDOM_STATE
    assert split["stratify"] == "high_traffic"
    assert split["final_test_usage"] == "untouched_until_final_evaluation"
    assert preprocessing["numeric_scaler"] == "StandardScaler"
