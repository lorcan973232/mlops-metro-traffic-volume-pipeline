"""Shared prediction schema for the traffic-volume API and UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data import FEATURE_COLUMNS, NUMERIC_FEATURES, WEATHER_MAIN_VALUES

TARGET_NAME = "high_traffic"
TARGET_LABEL = "Traffic Level"
TARGET_LABELS = {
    0: "normal traffic",
    1: "high traffic",
}

NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "temp": (230.0, 320.0),
    "rain_1h": (0.0, 60.0),
    "snow_1h": (0.0, 1.0),
    "clouds_all": (0.0, 100.0),
    "hour": (0.0, 23.0),
    "month": (1.0, 12.0),
    "day_of_week": (0.0, 6.0),
    "is_weekend": (0.0, 1.0),
    "is_holiday": (0.0, 1.0),
    "lag_1h_volume": (0.0, 8000.0),
    "lag_24h_volume": (0.0, 8000.0),
    "lag_168h_volume": (0.0, 8000.0),
    "rolling_3h_volume": (0.0, 8000.0),
    "rolling_24h_volume": (0.0, 8000.0),
}

WEATHER_OPTIONS = WEATHER_MAIN_VALUES

FEATURE_GROUPS = {
    "Date and Weather": [
        "hour",
        "month",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "weather_main",
        "temp",
        "clouds_all",
        "rain_1h",
        "snow_1h",
    ],
    "Recent Traffic": [
        "lag_1h_volume",
        "lag_24h_volume",
        "lag_168h_volume",
        "rolling_3h_volume",
        "rolling_24h_volume",
    ],
}


@dataclass(frozen=True)
class PredictionRequestExample:
    """Known-valid traffic sample used by smoke tests and the example button."""

    temp: float = 288.3
    rain_1h: float = 0.0
    snow_1h: float = 0.0
    clouds_all: float = 20.0
    hour: float = 8.0
    month: float = 10.0
    day_of_week: float = 1.0
    is_weekend: float = 0.0
    is_holiday: float = 0.0
    weather_main: str = "Clear"
    lag_1h_volume: float = 5200.0
    lag_24h_volume: float = 5000.0
    lag_168h_volume: float = 4800.0
    rolling_3h_volume: float = 4900.0
    rolling_24h_volume: float = 3300.0

    def as_payload(self) -> dict[str, dict[str, float | str]]:
        return {"features": self.__dict__.copy()}


def feature_label(feature_name: str) -> str:
    """Return the short label shown next to a feature in the browser form."""
    labels = {
        "temp": "Temperature (K)",
        "rain_1h": "Rain Last Hour",
        "snow_1h": "Snow Last Hour",
        "clouds_all": "Cloud Cover (%)",
        "hour": "Hour",
        "month": "Month",
        "day_of_week": "Day of Week",
        "is_weekend": "Weekend",
        "is_holiday": "Holiday",
        "weather_main": "Weather",
        "lag_1h_volume": "Previous Hour Volume",
        "lag_24h_volume": "Same Hour Yesterday",
        "lag_168h_volume": "Same Hour Last Week",
        "rolling_3h_volume": "Recent 3h Average",
        "rolling_24h_volume": "Recent 24h Average",
    }
    return labels[feature_name]


def feature_helper(feature_name: str) -> str:
    """Return a short help sentence for the browser input field."""
    helpers = {
        "temp": "Kelvin value from the traffic weather record.",
        "rain_1h": "Millimetres of rain in the last hour.",
        "snow_1h": "Millimetres of snow in the last hour.",
        "clouds_all": "Cloud cover from 0 to 100.",
        "hour": "0 to 23.",
        "month": "1 to 12.",
        "day_of_week": "0 = Monday, 6 = Sunday.",
        "is_weekend": "0 = no, 1 = yes.",
        "is_holiday": "0 = no, 1 = yes.",
        "weather_main": "Main weather condition.",
        "lag_1h_volume": "Observed traffic count one hour earlier.",
        "lag_24h_volume": "Observed traffic count 24 hours earlier.",
        "lag_168h_volume": "Observed traffic count one week earlier.",
        "rolling_3h_volume": "Average observed count over the previous 3 hours.",
        "rolling_24h_volume": "Average observed count over the previous 24 hours.",
    }
    return helpers[feature_name]


def ui_feature_groups() -> list[dict[str, Any]]:
    """Build grouped field metadata so the UI and API schema stay aligned."""
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
                        "kind": "select" if feature_name == "weather_main" else "number",
                        "options": WEATHER_OPTIONS if feature_name == "weather_main" else [],
                        "min": NUMERIC_RANGES.get(feature_name, (None, None))[0],
                        "max": NUMERIC_RANGES.get(feature_name, (None, None))[1],
                    }
                    for feature_name in features
                ],
            }
        )
    return grouped_features


def validate_prediction_payload(payload: object) -> list[dict[str, float | str]]:
    """Check API input and return records in the exact model feature order.

    The API accepts either one feature object, a `features` wrapper, or a list of
    objects. The final records are ordered by `FEATURE_COLUMNS` so training and
    prediction use the same schema.
    """
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
    clean_records: list[dict[str, float | str]] = []
    for record in records:
        unknown = sorted(set(record) - allowed_columns)
        if unknown:
            raise ValueError(f"Unknown feature columns: {unknown}")
        missing = [column for column in FEATURE_COLUMNS if column not in record]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        clean_record: dict[str, float | str] = {}
        weather = str(record["weather_main"])
        if weather not in WEATHER_OPTIONS:
            raise ValueError(f"Feature 'weather_main' must be one of: {WEATHER_OPTIONS}.")
        clean_record["weather_main"] = weather

        for column in NUMERIC_FEATURES:
            try:
                value = float(record[column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature '{column}' must be numeric.") from exc
            minimum, maximum = NUMERIC_RANGES[column]
            if value < minimum or value > maximum:
                raise ValueError(f"Feature '{column}' must be between {minimum} and {maximum}.")
            clean_record[column] = value
        clean_records.append({column: clean_record[column] for column in FEATURE_COLUMNS})

    return clean_records
