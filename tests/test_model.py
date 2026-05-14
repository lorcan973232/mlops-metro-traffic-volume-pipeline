from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from src.data import FEATURE_COLUMNS
from src.train import MODEL_HYPERPARAMETERS, RANDOM_STATE, TEST_SIZE, build_pipeline


def sample_training_frame() -> tuple[pd.DataFrame, np.ndarray]:
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.drop(columns=["target"]).copy()
    frame.columns = [column.strip().replace(" ", "_") for column in frame.columns]
    labels = np.where(dataset.target.to_numpy() == 0, "malignant", "benign")
    sample_index = list(np.where(labels == "malignant")[0][:20]) + list(
        np.where(labels == "benign")[0][:20]
    )
    return frame[FEATURE_COLUMNS].iloc[sample_index], labels[sample_index]


def test_model_pipeline_fits_and_predicts_with_selected_schema() -> None:
    frame, labels = sample_training_frame()
    pipeline = build_pipeline()
    pipeline.fit(frame, labels)
    predictions = pipeline.predict(frame)
    assert len(predictions) == len(frame)
    assert set(predictions).issubset({"malignant", "benign"})


def test_model_hyperparameters_are_explicit_and_reproducible() -> None:
    classifier = MODEL_HYPERPARAMETERS["classifier"]
    split = MODEL_HYPERPARAMETERS["train_test_split"]
    preprocessing = MODEL_HYPERPARAMETERS["preprocessing"]

    assert MODEL_HYPERPARAMETERS["algorithm"] == "LogisticRegression"
    assert classifier["C"] == 0.3
    assert classifier["class_weight"] == "balanced"
    assert classifier["max_iter"] == 2000
    assert classifier["penalty"] == "l2"
    assert classifier["random_state"] == RANDOM_STATE
    assert classifier["solver"] == "lbfgs"
    assert split["test_size"] == TEST_SIZE
    assert split["random_state"] == RANDOM_STATE
    assert split["stratify"] == "diagnosis_class"
    assert preprocessing["numeric_imputer_strategy"] == "median"
    assert preprocessing["numeric_scaler"] == "RobustScaler"
