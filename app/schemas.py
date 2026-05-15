from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FEATURE_COLUMNS = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",
    "radius_error",
    "texture_error",
    "perimeter_error",
    "area_error",
    "smoothness_error",
    "compactness_error",
    "concavity_error",
    "concave_points_error",
    "symmetry_error",
    "fractal_dimension_error",
    "worst_radius",
    "worst_texture",
    "worst_perimeter",
    "worst_area",
    "worst_smoothness",
    "worst_compactness",
    "worst_concavity",
    "worst_concave_points",
    "worst_symmetry",
    "worst_fractal_dimension",
]

CLASS_LABELS = ("malignant", "benign")

FEATURE_GROUPS = {
    "Mean Measurements": [
        "mean_radius",
        "mean_texture",
        "mean_perimeter",
        "mean_area",
        "mean_smoothness",
        "mean_compactness",
        "mean_concavity",
        "mean_concave_points",
        "mean_symmetry",
        "mean_fractal_dimension",
    ],
    "Standard Error Measurements": [
        "radius_error",
        "texture_error",
        "perimeter_error",
        "area_error",
        "smoothness_error",
        "compactness_error",
        "concavity_error",
        "concave_points_error",
        "symmetry_error",
        "fractal_dimension_error",
    ],
    "Worst Measurements": [
        "worst_radius",
        "worst_texture",
        "worst_perimeter",
        "worst_area",
        "worst_smoothness",
        "worst_compactness",
        "worst_concavity",
        "worst_concave_points",
        "worst_symmetry",
        "worst_fractal_dimension",
    ],
}


@dataclass(frozen=True)
class PredictionRequestExample:
    mean_radius: float = 17.99
    mean_texture: float = 10.38
    mean_perimeter: float = 122.8
    mean_area: float = 1001.0
    mean_smoothness: float = 0.1184
    mean_compactness: float = 0.2776
    mean_concavity: float = 0.3001
    mean_concave_points: float = 0.1471
    mean_symmetry: float = 0.2419
    mean_fractal_dimension: float = 0.07871
    radius_error: float = 1.095
    texture_error: float = 0.9053
    perimeter_error: float = 8.589
    area_error: float = 153.4
    smoothness_error: float = 0.006399
    compactness_error: float = 0.04904
    concavity_error: float = 0.05373
    concave_points_error: float = 0.01587
    symmetry_error: float = 0.03003
    fractal_dimension_error: float = 0.006193
    worst_radius: float = 25.38
    worst_texture: float = 17.33
    worst_perimeter: float = 184.6
    worst_area: float = 2019.0
    worst_smoothness: float = 0.1622
    worst_compactness: float = 0.6656
    worst_concavity: float = 0.7119
    worst_concave_points: float = 0.2654
    worst_symmetry: float = 0.4601
    worst_fractal_dimension: float = 0.1189

    def as_payload(self) -> dict[str, dict[str, float]]:
        return {"features": self.__dict__.copy()}


def feature_label(feature_name: str) -> str:
    return feature_name.replace("_", " ").title()


def feature_helper(feature_name: str) -> str:
    if feature_name.startswith("mean_"):
        return "Mean value from the diagnostic cell-nuclei measurements."
    if feature_name.endswith("_error"):
        return "Standard error measurement from the diagnostic sample."
    if feature_name.startswith("worst_"):
        return "Largest or most severe measurement observed in the sample."
    return "Numeric diagnostic feature used by the trained model."


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
