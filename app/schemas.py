from __future__ import annotations

from dataclasses import dataclass

FEATURE_COLUMNS = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

CLASS_LABELS = ("low", "medium", "high")


@dataclass(frozen=True)
class PredictionRequestExample:
    fixed_acidity: float = 7.0
    volatile_acidity: float = 0.27
    citric_acid: float = 0.36
    residual_sugar: float = 20.7
    chlorides: float = 0.045
    free_sulfur_dioxide: float = 45.0
    total_sulfur_dioxide: float = 170.0
    density: float = 1.001
    pH: float = 3.0
    sulphates: float = 0.45
    alcohol: float = 8.8

    def as_payload(self) -> dict[str, dict[str, float]]:
        return {"features": self.__dict__.copy()}


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
