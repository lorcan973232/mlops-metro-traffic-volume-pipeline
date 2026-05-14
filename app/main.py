from __future__ import annotations

from typing import Any

import pandas as pd
from flask import Flask, jsonify, request

from app.model_loader import load_model
from app.schemas import FEATURE_COLUMNS, validate_prediction_payload


def create_app(model_bundle: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MODEL_BUNDLE"] = model_bundle

    def get_model_bundle() -> dict[str, Any]:
        if app.config["MODEL_BUNDLE"] is None:
            app.config["MODEL_BUNDLE"] = load_model()
        return app.config["MODEL_BUNDLE"]

    @app.get("/health")
    def health() -> tuple[Any, int]:
        try:
            bundle = get_model_bundle()
        except Exception as exc:  # pragma: no cover - exercised by smoke tests
            return jsonify({"status": "unhealthy", "error": str(exc)}), 503

        return (
            jsonify(
                {
                    "status": "healthy",
                    "model_loaded": True,
                    "model_version": bundle.get("model_version", "unknown"),
                    "model_path": bundle.get(
                        "model_path",
                        "models/breast_cancer_classifier.joblib",
                    ),
                    "dataset": bundle.get("dataset", {}),
                    "feature_count": len(bundle["feature_columns"]),
                    "class_labels": list(bundle["class_labels"]),
                }
            ),
            200,
        )

    @app.post("/predict")
    def predict() -> tuple[Any, int]:
        try:
            records = validate_prediction_payload(request.get_json(force=True))
            frame = pd.DataFrame(records, columns=FEATURE_COLUMNS)
            bundle = get_model_bundle()
            model = bundle["model"]
            predictions = model.predict(frame)
            probabilities = model.predict_proba(frame)
            classes = list(model.classes_)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - exercised by smoke tests
            return jsonify({"error": str(exc)}), 500

        results = []
        for row_index, prediction in enumerate(predictions):
            results.append(
                {
                    "prediction": str(prediction),
                    "probabilities": {
                        str(label): float(probabilities[row_index][class_index])
                        for class_index, label in enumerate(classes)
                    },
                }
            )

        response: dict[str, Any] = {"predictions": results}
        response["model_version"] = bundle.get("model_version", "unknown")
        if len(results) == 1:
            response.update(results[0])
        return jsonify(response), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
