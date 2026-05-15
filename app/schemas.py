from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FEATURE_COLUMNS = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "orientation",
    "glazing_area",
    "glazing_area_distribution",
]

TARGET_NAME = "heating_load"
TARGET_LABEL = "Heating Load"
TARGET_UNIT = "load units"

FEATURE_OPTIONS: dict[str, list[dict[str, float | str]]] = {
    "orientation": [
        {"value": 2, "label": "2 - North"},
        {"value": 3, "label": "3 - East"},
        {"value": 4, "label": "4 - South"},
        {"value": 5, "label": "5 - West"},
    ],
    "glazing_area": [
        {"value": 0.0, "label": "0.00 - No glazing"},
        {"value": 0.1, "label": "0.10 - Low glazing"},
        {"value": 0.25, "label": "0.25 - Medium glazing"},
        {"value": 0.4, "label": "0.40 - High glazing"},
    ],
    "glazing_area_distribution": [
        {"value": 0, "label": "0 - No glazing"},
        {"value": 1, "label": "1 - Uniform"},
        {"value": 2, "label": "2 - North-facing"},
        {"value": 3, "label": "3 - East-facing"},
        {"value": 4, "label": "4 - South-facing"},
        {"value": 5, "label": "5 - West-facing"},
    ],
}

FEATURE_GROUPS = {
    "Building Shape": [
        "relative_compactness",
        "surface_area",
        "wall_area",
        "roof_area",
    ],
    "Building Setup": [
        "overall_height",
        "orientation",
        "glazing_area",
        "glazing_area_distribution",
    ],
}


@dataclass(frozen=True)
class PredictionRequestExample:
    relative_compactness: float = 0.76
    surface_area: float = 661.5
    wall_area: float = 416.5
    roof_area: float = 122.5
    overall_height: float = 7.0
    orientation: float = 2.0
    glazing_area: float = 0.4
    glazing_area_distribution: float = 5.0

    def as_payload(self) -> dict[str, dict[str, float]]:
        return {"features": self.__dict__.copy()}


def feature_label(feature_name: str) -> str:
    labels = {
        "relative_compactness": "Relative Compactness",
        "surface_area": "Surface Area",
        "wall_area": "Wall Area",
        "roof_area": "Roof Area",
        "overall_height": "Overall Height",
        "orientation": "Orientation",
        "glazing_area": "Glazing Area",
        "glazing_area_distribution": "Glazing Distribution",
    }
    return labels[feature_name]


def feature_helper(feature_name: str) -> str:
    helpers = {
        "relative_compactness": "Building compactness score, dataset range 0.62 to 0.98.",
        "surface_area": "Total external surface area, dataset range 514.5 to 808.5.",
        "wall_area": "Wall area, dataset range 245.0 to 416.5.",
        "roof_area": "Roof area, dataset range 110.25 to 220.5.",
        "overall_height": "Building height, usually 3.5 or 7.0.",
        "orientation": "Integer-coded building orientation from the UCI dataset.",
        "glazing_area": "Window/glazing proportion from 0.0 to 0.4.",
        "glazing_area_distribution": "Integer-coded window distribution from 0 to 5.",
    }
    return helpers[feature_name]


def ui_feature_groups() -> list[dict[str, Any]]:
    example = PredictionRequestExample().__dict__
    grouped_features = []
    for group_name, features in FEATURE_GROUPS.items():
        grouped_features.append(
            {
                "name": group_name,
                "features": [
                    {
                        "name": feature_name,
                        "label": feature_label(feature_name),
                        "helper": feature_helper(feature_name),
                        "example": example[feature_name],
                        "options": FEATURE_OPTIONS.get(feature_name),
                    }
                    for feature_name in features
                ],
            }
        )
    return grouped_features


def validate_prediction_payload(payload: object) -> list[dict[str, float]]:
    if isinstance(payload, dict) and "features" in payload:
        payload = payload["features"]

    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list) and all(isinstance(record, dict) for record in payload):
        records = payload
    else:
        raise ValueError(
            "Payload must be a feature object, {'features': object}, or list of objects."
        )

    allowed_columns = set(FEATURE_COLUMNS)
    clean_records: list[dict[str, float]] = []
    for record in records:
        unknown = sorted(set(record) - allowed_columns)
        if unknown:
            raise ValueError(f"Unknown feature columns: {unknown}")

        missing = [column for column in FEATURE_COLUMNS if column not in record]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        clean_record: dict[str, float] = {}
        for column in FEATURE_COLUMNS:
            try:
                clean_record[column] = float(record[column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature '{column}' must be numeric.") from exc
        clean_records.append(clean_record)

    return clean_records
