"""Train and evaluate the reproducible majority-class baseline."""

from __future__ import annotations

import json

import pandas as pd

from src.evaluate import BASELINE_METRICS_PATH, _evaluate_baseline
from src.train import load_processed_data, split_train_validation_test


def run_baseline() -> dict:
    """Evaluate the baseline on the untouched final test split."""
    data = load_processed_data()
    x_train, x_validation, x_test, y_train, y_validation, y_test = (
        split_train_validation_test(data)
    )
    x_train_validation = pd.concat([x_train, x_validation], axis=0)
    y_train_validation = pd.concat([y_train, y_validation], axis=0)
    baseline = _evaluate_baseline(x_train_validation, y_train_validation, x_test, y_test)
    baseline["created_by"] = "python -m src.baseline"
    BASELINE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_METRICS_PATH.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return baseline


def main() -> None:
    """CLI entry point for explicit baseline reproduction."""
    print(json.dumps(run_baseline(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
