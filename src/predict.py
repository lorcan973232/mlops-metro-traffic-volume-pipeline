from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from app.schemas import FEATURE_COLUMNS, PredictionRequestExample, validate_prediction_payload
from src.train import MODEL_PATH


def predict(payload: dict, model_path: Path = MODEL_PATH) -> list[dict[str, object]]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run `python -m src.train` first."
        )
    bundle = joblib.load(model_path)
    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Saved model feature schema does not match prediction schema.")
    records = validate_prediction_payload(payload)
    frame = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    model = bundle["model"]
    predictions = model.predict(frame)
    probabilities = model.predict_proba(frame)
    classes = list(model.classes_)
    return [
        {
            "prediction": str(prediction),
            "probabilities": {
                str(label): float(probabilities[row_index][class_index])
                for class_index, label in enumerate(classes)
            },
        }
        for row_index, prediction in enumerate(predictions)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one local breast cancer diagnosis prediction."
    )
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
