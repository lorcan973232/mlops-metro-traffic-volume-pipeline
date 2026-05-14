from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib

from app.schemas import CLASS_LABELS, FEATURE_COLUMNS

DEFAULT_MODEL_PATH = Path("models/breast_cancer_classifier.joblib")


def load_model(model_path: str | Path | None = None) -> dict[str, Any]:
    resolved_path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model artefact not found at {resolved_path}. Run `python -m src.train` first."
        )
    bundle = joblib.load(resolved_path)
    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError(
            "Model feature schema is incompatible with the API prediction schema."
        )
    if tuple(bundle.get("class_labels", ())) != CLASS_LABELS:
        raise ValueError("Model class labels are incompatible with the API schema.")
    return bundle
