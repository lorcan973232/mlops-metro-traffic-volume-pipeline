"""Load the saved model bundle for Flask, tests, Docker, and Kind.

This is a small file, but it protects an important contract: the serving layer
must only use a model whose feature schema and task type match the API request
schema. If that check were missing, a model trained on different columns could be
served without an obvious startup failure.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.schemas import FEATURE_COLUMNS
from src.sklearn_compat import load_joblib_bundle

DEFAULT_MODEL_PATH = Path("models/wine_quality_classifier.joblib")


def load_model(model_path: str | Path | None = None) -> dict[str, Any]:
    """Load the saved model bundle and check it matches the API schema.

    `MODEL_PATH` lets Docker or a local demo point at a different saved bundle,
    but the schema check still has to pass before predictions are served.
    """
    # Resolve the path at request/startup time so local runs, Docker, and Kind
    # can all use the same code while still allowing `MODEL_PATH` to be set by
    # the container environment.
    resolved_path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model artefact not found at {resolved_path}. Run `python -m src.train` first."
        )
    # The joblib bundle comes from `src.train`. Loading it here joins the training
    # side of the pipeline to the serving side used by Flask and smoke tests.
    bundle = load_joblib_bundle(resolved_path)
    # The feature order is part of the model contract. If the API accepted a
    # different order, predictions could be wrong without raising an obvious error.
    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Model feature schema is incompatible with the API prediction schema.")
    # The API is written for binary classification. This guard stops a different
    # task type from being served through the same `/predict` response shape.
    if bundle.get("task_type") != "classification":
        raise ValueError("Model task type is incompatible with the classification API schema.")
    return bundle
