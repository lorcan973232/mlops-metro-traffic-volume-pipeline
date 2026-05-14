from __future__ import annotations

import pandas as pd

from src.data import FEATURE_COLUMNS
from src.train import MODEL_HYPERPARAMETERS, RANDOM_STATE, TEST_SIZE, build_pipeline


def test_model_pipeline_fits_and_predicts_with_selected_schema() -> None:
    frame = pd.DataFrame(
        [
            [7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8],
            [6.3, 0.30, 0.34, 1.6, 0.049, 14.0, 132.0, 0.994, 3.3, 0.49, 9.5],
            [8.1, 0.28, 0.40, 6.9, 0.050, 30.0, 97.0, 0.995, 3.2, 0.44, 10.1],
            [6.9, 0.22, 0.43, 2.1, 0.040, 18.0, 112.0, 0.992, 3.1, 0.53, 11.4],
            [7.4, 0.18, 0.31, 8.4, 0.041, 44.0, 124.0, 0.993, 3.0, 0.60, 12.0],
            [5.9, 0.42, 0.20, 1.8, 0.060, 12.0, 90.0, 0.997, 3.5, 0.38, 8.4],
        ],
        columns=FEATURE_COLUMNS,
    )
    labels = ["medium", "low", "medium", "high", "high", "low"]
    pipeline = build_pipeline()
    pipeline.fit(frame, labels)
    predictions = pipeline.predict(frame)
    assert len(predictions) == len(frame)
    assert set(predictions).issubset({"low", "medium", "high"})


def test_model_hyperparameters_are_explicit_and_reproducible() -> None:
    classifier = MODEL_HYPERPARAMETERS["classifier"]
    split = MODEL_HYPERPARAMETERS["train_test_split"]
    preprocessing = MODEL_HYPERPARAMETERS["preprocessing"]

    assert MODEL_HYPERPARAMETERS["algorithm"] == "RandomForestClassifier"
    assert classifier["n_estimators"] == 120
    assert classifier["max_depth"] == 12
    assert classifier["min_samples_leaf"] == 2
    assert classifier["class_weight"] == "balanced_subsample"
    assert classifier["random_state"] == RANDOM_STATE
    assert split["test_size"] == TEST_SIZE
    assert split["random_state"] == RANDOM_STATE
    assert split["stratify"] == "quality_class"
    assert preprocessing["numeric_imputer_strategy"] == "median"
    assert preprocessing["numeric_scaler"] == "StandardScaler"
