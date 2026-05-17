from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FEATURE_COLUMNS = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]

TARGET_NAME = "quality_label"
TARGET_LABEL = "Wine Quality Class"
TARGET_LABELS = {
    0: "standard quality",
    1: "good quality",
}

FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "fixed_acidity": (4.0, 16.0),
    "volatile_acidity": (0.1, 1.6),
    "citric_acid": (0.0, 1.1),
    "residual_sugar": (0.5, 16.0),
    "chlorides": (0.01, 0.7),
    "free_sulfur_dioxide": (1.0, 80.0),
    "total_sulfur_dioxide": (5.0, 300.0),
    "density": (0.98, 1.01),
    "ph": (2.5, 4.2),
    "sulphates": (0.2, 2.2),
    "alcohol": (8.0, 16.0),
}

# These groups are only for the browser form. They make the live demo easier to
# explain without changing the feature order used by the trained model.
FEATURE_GROUPS = {
    "Acidity Profile": [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "ph",
    ],
    "Fermentation Chemistry": [
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
    ],
    "Body and Finish": [
        "density",
        "sulphates",
        "alcohol",
    ],
}


@dataclass(frozen=True)
class PredictionRequestExample:
    fixed_acidity: float = 7.4
    volatile_acidity: float = 0.70
    citric_acid: float = 0.00
    residual_sugar: float = 1.9
    chlorides: float = 0.076
    free_sulfur_dioxide: float = 11.0
    total_sulfur_dioxide: float = 34.0
    density: float = 0.9978
    ph: float = 3.51
    sulphates: float = 0.56
    alcohol: float = 9.4

    def as_payload(self) -> dict[str, dict[str, float]]:
        return {"features": self.__dict__.copy()}


def feature_label(feature_name: str) -> str:
    labels = {
        "fixed_acidity": "Fixed Acidity",
        "volatile_acidity": "Volatile Acidity",
        "citric_acid": "Citric Acid",
        "residual_sugar": "Residual Sugar",
        "chlorides": "Chlorides",
        "free_sulfur_dioxide": "Free SO2",
        "total_sulfur_dioxide": "Total SO2",
        "density": "Density",
        "ph": "pH",
        "sulphates": "Sulphates",
        "alcohol": "Alcohol",
    }
    return labels[feature_name]


def feature_helper(feature_name: str) -> str:
    helpers = {
        "fixed_acidity": "Non-volatile tartaric acid concentration.",
        "volatile_acidity": "Acetic acid level; high values can reduce quality.",
        "citric_acid": "Citric acid concentration.",
        "residual_sugar": "Sugar left after fermentation.",
        "chlorides": "Salt concentration in the wine.",
        "free_sulfur_dioxide": "Free sulfur dioxide concentration.",
        "total_sulfur_dioxide": "Total sulfur dioxide concentration.",
        "density": "Wine density, usually near 1.0.",
        "ph": "Acidity/alkalinity value.",
        "sulphates": "Sulphate concentration.",
        "alcohol": "Alcohol by volume percentage.",
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
                        "min": FEATURE_RANGES[feature_name][0],
                        "max": FEATURE_RANGES[feature_name][1],
                    }
                    for feature_name in features
                ],
            }
        )
    return grouped_features


def validate_prediction_payload(payload: object) -> list[dict[str, float]]:
    """Validate user input before it reaches the model prediction pipeline."""
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
            # Range checks catch obvious input mistakes during the API smoke tests
            # and live demo, while still using broad bounds from the source data.
            try:
                value = float(record[column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature '{column}' must be numeric.") from exc
            minimum, maximum = FEATURE_RANGES[column]
            if value < minimum or value > maximum:
                raise ValueError(
                    f"Feature '{column}' must be between {minimum} and {maximum}."
                )
            clean_record[column] = value
        clean_records.append(clean_record)

    return clean_records
