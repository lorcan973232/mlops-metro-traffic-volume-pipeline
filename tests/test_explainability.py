from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_explainability_script_generates_model_derived_reports() -> None:
    env = {**os.environ, "FAST_MODE": "1"}
    required = [
        Path("reports/explainability/shap_summary.json"),
        Path("reports/explainability/shap_feature_importance.json"),
        Path("reports/explainability/local_explanation_example.json"),
    ]
    backups = {
        path: path.read_bytes() if path.exists() else None
        for path in required
    }
    try:
        subprocess.run([sys.executable, "scripts/explain_model.py"], check=True, env=env)
        for path in required:
            assert path.is_file()

        summary = json.loads(required[0].read_text(encoding="utf-8"))
        importance = json.loads(required[1].read_text(encoding="utf-8"))
        local = json.loads(required[2].read_text(encoding="utf-8"))
    finally:
        for path, content in backups.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)

    assert summary["status"] in {"computed", "shap_fallback"}
    assert summary["method"] in {"shap.TreeExplainer", "sklearn.permutation_importance"}
    assert summary["computed_from_model"] is True
    assert len(summary["top_features"]) >= 3
    assert "placeholder" not in json.dumps(summary).lower()
    assert importance["computed_from_model"] is True
    assert set(importance["feature_importance"])
    assert local["computed_from_model"] is True
    assert local["prediction"]["prediction"] in {0, 1}
