from __future__ import annotations

from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template, request

from app.model_loader import load_model
from app.schemas import (
    CLASS_LABELS,
    FEATURE_COLUMNS,
    PredictionRequestExample,
    ui_feature_groups,
    validate_prediction_payload,
)


def create_app(model_bundle: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MODEL_BUNDLE"] = model_bundle

    def get_model_bundle() -> dict[str, Any]:
        if app.config["MODEL_BUNDLE"] is None:
            app.config["MODEL_BUNDLE"] = load_model()
        return app.config["MODEL_BUNDLE"]

    @app.get("/")
    def index() -> str:
        model_status: dict[str, Any] = {
            "model_loaded": False,
            "model_version": "unavailable",
            "dataset": {"name": "UCI Breast Cancer Wisconsin Diagnostic"},
            "class_labels": list(CLASS_LABELS),
        }
        try:
            bundle = get_model_bundle()
            model_status.update(
                {
                    "model_loaded": True,
                    "model_version": bundle.get("model_version", "unknown"),
                    "model_path": bundle.get(
                        "model_path",
                        "models/breast_cancer_classifier.joblib",
                    ),
                    "dataset": bundle.get("dataset", model_status["dataset"]),
                    "class_labels": list(bundle.get("class_labels", CLASS_LABELS)),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive UI status path
            model_status["error"] = str(exc)

        return render_template(
            "index.html",
            feature_groups=ui_feature_groups(),
            feature_columns=FEATURE_COLUMNS,
            example_payload=PredictionRequestExample().as_payload()["features"],
            model_status=model_status,
        )

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
    app.run(host="0.0.0.0", port=5000)
