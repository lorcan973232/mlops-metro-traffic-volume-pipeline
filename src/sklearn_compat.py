"""Compatibility helpers for loading the committed scikit-learn model.

The saved model was trained and written with scikit-learn 1.5.1, which is the
version pinned in `requirements.txt`. Some local machines may still run tests
with a newer system Python environment. This file keeps model loading explicit:
it installs the one missing unpickle helper needed by newer scikit-learn versions
without changing the trained model, feature schema, metrics, or predictions.
"""

from __future__ import annotations

import warnings
from collections import UserList
from pathlib import Path
from typing import Any

import joblib


def install_column_transformer_remainder_shim() -> None:
    """Install the old `ColumnTransformer` remainder-list class when missing.

    Joblib needs this class name while reading model files created by
    scikit-learn 1.5.1. Newer scikit-learn releases removed the private helper,
    so the unpickler fails before it can reach the actual estimator. The shim is
    only used when the class is absent, and it behaves like the old list wrapper.
    """
    import sklearn.compose._column_transformer as column_transformer

    if hasattr(column_transformer, "_RemainderColsList"):
        return

    class _RemainderColsList(UserList):
        """Small copy of the old scikit-learn remainder-column list wrapper."""

        def __init__(
            self,
            columns: list[int],
            *,
            future_dtype: str | None = None,
            warning_was_emitted: bool = False,
            warning_enabled: bool = True,
        ) -> None:
            """Store the old remainder-column metadata expected by joblib."""
            super().__init__(columns)
            self.future_dtype = future_dtype
            self.warning_was_emitted = warning_was_emitted
            self.warning_enabled = warning_enabled

        def __getitem__(self, index: Any) -> Any:
            self._show_remainder_cols_warning()
            return super().__getitem__(index)

        def _show_remainder_cols_warning(self) -> None:
            if self.warning_was_emitted or not self.warning_enabled:
                return
            self.warning_was_emitted = True
            warnings.warn(
                "Using a compatibility list for old scikit-learn ColumnTransformer "
                "remainder columns while loading the saved project model.",
                FutureWarning,
                stacklevel=2,
            )

    column_transformer._RemainderColsList = _RemainderColsList


def load_joblib_bundle(path: str | Path) -> dict[str, Any]:
    """Load a saved joblib model bundle after installing compatibility helpers."""
    install_column_transformer_remainder_shim()
    bundle = joblib.load(path)
    _patch_loaded_estimators(bundle)
    return bundle


def _patch_loaded_estimators(value: Any) -> None:
    """Patch old estimator attributes expected by newer scikit-learn at runtime.

    The project still pins scikit-learn 1.5.1 for normal use. This recursive pass
    only helps when a newer system environment runs tests against the committed
    model. It does not retrain the model or change learned values.
    """
    from sklearn.impute import SimpleImputer

    if isinstance(value, SimpleImputer) and not hasattr(value, "_fill_dtype"):
        value._fill_dtype = getattr(value, "_fit_dtype", None)

    if isinstance(value, dict):
        for item in value.values():
            _patch_loaded_estimators(item)
        return

    for attribute in ("steps", "transformers_", "estimators_"):
        for item in getattr(value, attribute, []) or []:
            if isinstance(item, tuple):
                for part in item:
                    _patch_loaded_estimators(part)
            else:
                _patch_loaded_estimators(item)
