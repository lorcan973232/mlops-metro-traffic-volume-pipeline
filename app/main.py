"""Flask API and browser UI entry point for the trained traffic-volume model.

The app is used locally, inside Docker, and inside the Kind deployment. Tests can
inject a model bundle, but normal runs load the same saved artefact produced by
`src.train`. The route design is intentionally small: `/health` checks the model
is loaded, `/predict` validates and scores real feature payloads, and `/` serves
the browser form that calls the same API.
"""

from __future__ import annotations

import json
import logging
import traceback
import uuid
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template, request

from app.dashboard import dashboard_bp
from app.model_loader import load_model
from app.schemas import (
    FEATURE_COLUMNS,
    TARGET_LABEL,
    TARGET_LABELS,
    TARGET_NAME,
    PredictionRequestExample,
    ui_feature_groups,
    validate_prediction_payload,
)

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# Flask application
# ==============================================================================
#
# The UI and API share the same `/predict` route. A browser prediction exercises
# the same validation and model bundle as the smoke tests and Docker/Kind
# deployments.


def _target_labels(bundle: dict[str, Any]) -> dict[int, str]:
    """Normalise saved target-label keys so JSON and joblib metadata agree."""
    return {int(key): value for key, value in bundle.get("target_labels", TARGET_LABELS).items()}


def create_app(model_bundle: dict[str, Any] | None = None) -> Flask:
    """Create the Flask app, with optional model injection for tests."""
    app = Flask(__name__)
    app.config["MODEL_BUNDLE"] = model_bundle

    def get_model_bundle() -> dict[str, Any]:
        """Load the model once per app instance so requests reuse the same model file."""
        if app.config["MODEL_BUNDLE"] is None:
            app.config["MODEL_BUNDLE"] = load_model()
        return app.config["MODEL_BUNDLE"]

    @app.get("/")
    def index() -> str:
        model_status: dict[str, Any] = {
            "model_loaded": False,
            "model_version": "unavailable",
            "dataset": {"name": "UCI Metro Interstate Traffic Volume"},
            "target_label": TARGET_LABEL,
        }
        try:
            bundle = get_model_bundle()
            model_status.update(
                {
                    "model_loaded": True,
                    "model_version": bundle.get("model_version", "unknown"),
                    "model_path": bundle.get(
                        "model_path",
                        "models/traffic_volume_classifier.joblib",
                    ),
                    "dataset": bundle.get("dataset", model_status["dataset"]),
                    "target_labels": _target_labels(bundle),
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
                        "models/traffic_volume_classifier.joblib",
                    ),
                    "dataset": bundle.get("dataset", {}),
                    "feature_count": len(bundle["feature_columns"]),
                    "task_type": bundle.get("task_type", "classification"),
                    "target": bundle.get("target_definition", {}).get("model_target", TARGET_NAME),
                    "target_label": TARGET_LABEL,
                    "target_labels": _target_labels(bundle),
                    "classes": bundle.get("classes", [0, 1]),
                }
            ),
            200,
        )

    @app.post("/predict")
    def predict() -> tuple[Any, int]:
        request_id = str(uuid.uuid4())[:12]
        start_time = datetime.now(UTC)
        logger.info(
            json.dumps(
                {
                    "event": "prediction_request_started",
                    "request_id": request_id,
                    "timestamp": start_time.isoformat(),
                }
            )
        )
        try:
            records = validate_prediction_payload(request.get_json(force=True))
            # Build the DataFrame with the training feature order. The schema
            # validator has already checked names and values, but the model still
            # expects a consistent column order.
            frame = pd.DataFrame(records, columns=FEATURE_COLUMNS)
            bundle = get_model_bundle()
            model = bundle["model"]
            predictions = [int(value) for value in model.predict(frame)]
            probability_rows = (
                model.predict_proba(frame)
                if hasattr(model, "predict_proba")
                else [None] * len(predictions)
            )
            execution_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            logger.info(
                json.dumps(
                    {
                        "event": "predictions_computed",
                        "request_id": request_id,
                        "prediction_count": len(predictions),
                        "execution_time_ms": round(execution_time_ms, 2),
                        "model_version": bundle.get("model_version", "unknown"),
                    }
                )
            )
        except ValueError as exc:
            logger.warning(
                json.dumps(
                    {
                        "event": "validation_error",
                        "request_id": request_id,
                        "error_type": "ValueError",
                        "error_message": str(exc),
                    }
                )
            )
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            logger.error(
                json.dumps(
                    {
                        "event": "prediction_failed",
                        "request_id": request_id,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            )
            return jsonify({"error": str(exc)}), 500

        labels = _target_labels(bundle)
        classes = [int(value) for value in bundle.get("classes", [0, 1])]
        results = []
        for prediction, probabilities in zip(predictions, probability_rows, strict=False):
            result: dict[str, Any] = {
                "prediction": prediction,
                "prediction_label": labels.get(prediction, str(prediction)),
                "target": bundle.get("target_definition", {}).get("model_target", TARGET_NAME),
            }
            if probabilities is not None:
                probability_map = {
                    labels.get(class_value, str(class_value)): round(float(probability), 4)
                    for class_value, probability in zip(classes, probabilities, strict=False)
                }
                result["probabilities"] = probability_map
                result["confidence"] = probability_map[labels.get(prediction, str(prediction))]
            results.append(result)

        response: dict[str, Any] = {"predictions": results}
        response["model_version"] = bundle.get("model_version", "unknown")
        if len(results) == 1:
            response.update(results[0])

        latency_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
        logger.info(
            json.dumps(
                {
                    "event": "response_sent",
                    "request_id": request_id,
                    "status_code": 200,
                    "model_version": bundle.get("model_version", "unknown"),
                    "latency_ms": round(latency_ms, 2),
                    "prediction_count": len(results),
                }
            )
        )
        return jsonify(response), 200

    app.register_blueprint(dashboard_bp)
    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
