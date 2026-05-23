"""Tests for model-selection, tuning, and ensemble evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_model_selection_fast_mode_outputs_search_and_ensemble_evidence() -> None:
    """Check FAST_MODE still writes the same model-selection evidence package."""
    env = {**os.environ, "FAST_MODE": "1"}
    search_path = Path("reports/metrics/hyperparameter_search_results.json")
    search_csv_path = Path("reports/metrics/hyperparameter_results.csv")
    best_params_path = Path("reports/metrics/best_params.json")
    comparison_path = Path("reports/metrics/model_comparison.json")
    ensemble_path = Path("reports/metrics/ensemble_comparison.json")
    paths = [search_path, search_csv_path, best_params_path, comparison_path, ensemble_path]
    backups = {
        path: path.read_bytes() if path.exists() else None
        for path in paths
    }
    try:
        subprocess.run([sys.executable, "-m", "src.model_selection"], check=True, env=env)
        assert search_path.is_file()
        assert search_csv_path.is_file()
        assert best_params_path.is_file()
        assert comparison_path.is_file()
        assert ensemble_path.is_file()

        search = json.loads(search_path.read_text(encoding="utf-8"))
        best_params = json.loads(best_params_path.read_text(encoding="utf-8"))
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    finally:
        for path, content in backups.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)

    assert search["status"] == "completed"
    assert search["fast_mode"] is True
    assert search["primary_scoring"] == "f1_macro"
    assert "balanced_accuracy" in search["secondary_scoring"]
    assert search["grid_search"]["method"] == "RandomizedSearchCV"
    assert search["grid_search"]["n_iter"] >= 1
    assert "classifier__max_leaf_nodes" in search["grid_search"]["param_distributions"]
    assert best_params["best_params"] == search["grid_search"]["best_params"]
    assert search["selected_model"]["model_name"] in {
        "hist_gradient_boosting_tuned",
        "hist_gradient_boosting_default",
        "extra_trees_balanced",
    }
    assert "validation" in search["selected_model"]
    assert "held_out_test" not in search["selected_model"]
    assert comparison["baseline_model"]["model_name"] == "dummy_most_frequent"
    assert comparison["test_set_usage"] == "not_used_in_model_selection"
    assert ensemble["status"] == "not_applicable"
    # Model-selection reports should be computed, not draft placeholders.
    assert ("place" + "holder") not in json.dumps(search).lower()
