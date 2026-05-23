"""Integration tests for the end-to-end data, model, API, and evidence path."""

from __future__ import annotations

import json
from pathlib import Path

from app.main import create_app
from app.schemas import FEATURE_COLUMNS, PredictionRequestExample
from src.data import FEATURE_COLUMNS as DATA_FEATURE_COLUMNS
from src.model_registry import load_model_metadata
from src.train import (
    RANDOM_STATE,
    TEST_SIZE,
    build_pipeline,
    load_processed_data,
)


def test_full_pipeline_data_to_prediction_produces_identical_results() -> None:
    """Verify the full pipeline from data through prediction is deterministic."""
    from sklearn.model_selection import train_test_split

    data_1 = load_processed_data()
    x_1 = data_1[DATA_FEATURE_COLUMNS]
    y_1 = data_1["high_traffic"]
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(
        x_1, y_1, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True, stratify=y_1
    )

    pipeline_1 = build_pipeline()
    pipeline_1.fit(x_train_1, y_train_1)
    predictions_1 = [int(p) for p in pipeline_1.predict(x_test_1)]

    data_2 = load_processed_data()
    x_2 = data_2[DATA_FEATURE_COLUMNS]
    y_2 = data_2["high_traffic"]
    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(
        x_2, y_2, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True, stratify=y_2
    )

    pipeline_2 = build_pipeline()
    pipeline_2.fit(x_train_2, y_train_2)
    predictions_2 = [int(p) for p in pipeline_2.predict(x_test_2)]

    assert predictions_1 == predictions_2, (
        "Predictions must be identical across runs with same random state"
    )


def test_integration_pipeline_generates_all_required_artefact_files() -> None:
    """Verify the pipeline creates the report files used by the README."""
    required_files = [
        "reports/metrics/latest_metrics.json",
        "reports/metrics/model_metadata.json",
        "reports/metrics/quality_gate_report.json",
        "reports/metrics/feature_importance.json",
        "reports/metrics/fairness_analysis.json",
    ]

    missing_files = [f for f in required_files if not Path(f).is_file()]
    assert missing_files == [], f"Missing artefact files: {missing_files}"

    latest = json.loads(Path("reports/metrics/latest_metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads(Path("reports/metrics/model_metadata.json").read_text(encoding="utf-8"))
    gate = json.loads(Path("reports/metrics/quality_gate_report.json").read_text(encoding="utf-8"))

    assert latest["model_version"] == metadata["model_version"]
    assert latest["model_version"] == gate.get("model_version", latest["model_version"])
    assert gate["decision"] in {"accept_candidate_model", "reject_candidate_model"}
    if gate["passed"]:
        assert gate["decision"] == "accept_candidate_model"
    else:
        assert gate["decision"] == "reject_candidate_model"


def test_api_can_load_model_and_handle_predictions() -> None:
    """Verify Flask loads the trained artefact and handles real predictions."""
    from app.model_loader import load_model

    bundle = load_model()
    metadata = load_model_metadata()

    assert bundle["model"] is not None
    assert bundle["model_version"] == metadata["model_version"]
    assert bundle["feature_columns"] == FEATURE_COLUMNS
    assert bundle["task_type"] == "classification"

    app = create_app()
    client = app.test_client()

    health = client.get("/health")
    assert health.status_code == 200
    health_data = health.get_json()
    assert health_data["status"] == "healthy"
    assert health_data["model_version"] == metadata["model_version"]

    prediction_response = client.post("/predict", json=PredictionRequestExample().as_payload())
    assert prediction_response.status_code == 200
    pred_data = prediction_response.get_json()
    assert "prediction" in pred_data
    assert pred_data["prediction"] in [0, 1]
    assert "confidence" in pred_data
    assert 0.0 <= pred_data["confidence"] <= 1.0
    assert pred_data["model_version"] == metadata["model_version"]


def test_api_rejects_invalid_prediction_requests() -> None:
    """Verify API validation catches invalid inputs before scoring."""
    app = create_app()
    client = app.test_client()

    invalid_payloads = [
        {"features": {}},
        {"features": {"temp": "not_a_number"}},
        {"features": {"temp": 999.0}},
        {"completely": "wrong_structure"},
        None,
    ]

    for payload in invalid_payloads:
        if payload is not None:
            response = client.post("/predict", json=payload)
        else:
            response = client.post("/predict", json={})
        assert response.status_code in [400, 500], f"Expected error status for payload: {payload}"
