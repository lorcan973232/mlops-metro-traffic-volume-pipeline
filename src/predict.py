"""Command-line prediction helper using the same model contract as Flask.

The CLI is used by smoke tests and README reproduction commands. It validates the
payload with `app.schemas`, loads the saved joblib bundle, and returns labels,
probabilities, confidence, target name, and model version so local predictions
match the API response shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.schemas import FEATURE_COLUMNS, PredictionRequestExample, validate_prediction_payload
from src.sklearn_compat import load_joblib_bundle
from src.train import MODEL_PATH

# ==============================================================================
# Local prediction helper
# ==============================================================================
#
# The command-line prediction path uses the same schema validation and saved
# model bundle as the Flask API. This keeps local smoke tests and the browser
# demo tied to the real trained artefact instead of a separate example path.


def _prediction_result(
    bundle: dict[str, Any],
    prediction: int,
    probabilities: Any | None,
) -> dict[str, Any]:
    target_labels = {int(key): value for key, value in bundle.get("target_labels", {}).items()}
    label = target_labels.get(prediction, str(prediction))
    result: dict[str, Any] = {
        "prediction": prediction,
        "prediction_label": label,
        "target": bundle.get("target_definition", {}).get("model_target", "high_traffic"),
    }
    if probabilities is not None:
        classes = [int(value) for value in bundle.get("classes", [0, 1])]
        probability_map = {
            target_labels.get(class_value, str(class_value)): round(float(probability), 4)
            for class_value, probability in zip(classes, probabilities, strict=False)
        }
        result["probabilities"] = probability_map
        if prediction in classes:
            result["confidence"] = probability_map[target_labels.get(prediction, str(prediction))]
    return result


def predict(payload: dict, model_path: Path = MODEL_PATH) -> list[dict[str, Any]]:
    """Validate a payload and score it with the saved training artefact."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artefact not found at {model_path}. Run `python -m src.train` first."
        )
    bundle = load_joblib_bundle(model_path)
    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Saved model feature schema does not match prediction schema.")
    if bundle.get("task_type") != "classification":
        raise ValueError("Saved model task type does not match classification prediction schema.")
    records = validate_prediction_payload(payload)
    frame = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    model = bundle["model"]
    predictions = [int(value) for value in model.predict(frame)]
    probability_rows = (
        model.predict_proba(frame) if hasattr(model, "predict_proba") else [None] * len(frame)
    )
    return [
        _prediction_result(bundle, prediction, probabilities)
        for prediction, probabilities in zip(predictions, probability_rows, strict=False)
    ]


def main() -> None:
    """Run one prediction from supplied JSON or the shared example payload."""
    parser = argparse.ArgumentParser(description="Run one local traffic-volume prediction.")
    parser.add_argument("--payload-json", default=None)
    args = parser.parse_args()
    payload = (
        json.loads(args.payload_json)
        if args.payload_json
        else PredictionRequestExample().as_payload()
    )
    print(json.dumps({"predictions": predict(payload)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
