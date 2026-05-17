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
    comparison_path = Path("reports/metrics/model_comparison.json")
    ensemble_path = Path("reports/metrics/ensemble_comparison.json")
    paths = [search_path, comparison_path, ensemble_path]
    backups = {
        path: path.read_bytes() if path.exists() else None
        for path in paths
    }
    try:
        subprocess.run([sys.executable, "-m", "src.model_selection"], check=True, env=env)
        assert search_path.is_file()
        assert comparison_path.is_file()
        assert ensemble_path.is_file()

        search = json.loads(search_path.read_text(encoding="utf-8"))
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
    assert search["grid_search"]["method"] == "GridSearchCV"
    assert search["selected_model"]["model_name"] in {
        "extra_trees_tuned",
        "soft_voting_ensemble",
    }
    assert comparison["baseline_model"]["model_name"] == "dummy_most_frequent"
    assert comparison["tuned_model"]["model_name"] == "extra_trees_tuned"
    assert comparison["ensemble_model"]["model_name"] == "soft_voting_ensemble"
    assert ensemble["ensemble_type"] == "VotingClassifier"
    assert isinstance(ensemble["selected"], bool)
    # Model-selection evidence should be computed, not a draft marker.
    assert ("place" + "holder") not in json.dumps(search).lower()
