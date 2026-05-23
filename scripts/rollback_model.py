#!/usr/bin/env python
"""Point lightweight registry metadata back to a prior accepted model version.

This is a model-management support script for the repository-local registry. It
does not change the trained model itself; it updates metadata so the current
version points at an earlier quality-gate-approved record. A real deployment
would need additional operational controls, which this student project does
not claim to provide.
"""
from __future__ import annotations

import json
import sys

from src.model_registry import MODEL_REGISTRY_PATH
from src.versioning import VERSION_MANIFEST_PATH, get_rollback_candidates, get_version_record


def rollback_model(target_version: str) -> None:
    """Update registry files to refer to a prior accepted rollback candidate."""
    candidates = get_rollback_candidates()

    if target_version not in candidates:
        print(f"FAIL: Version {target_version} is not available for rollback.")
        print(f"Available versions: {', '.join(candidates)}")
        raise ValueError(f"Cannot rollback to version {target_version}")

    version_record = get_version_record(target_version)
    if not version_record:
        raise ValueError(f"Version record not found for {target_version}")

    # The registry record is the pointer used by this project. Updating it is
    # enough for this lightweight model-management scope; no external registry or
    # cloud service is being claimed.
    registry = {
        "model_version": target_version,
        "model_path": version_record["model_path"],
        "metrics_path": version_record.get("metadata_path"),
        "dataset_source": "UCI Metro Interstate Traffic Volume",
        "dataset_doi": "10.24432/C5PC84",
        "task_type": "classification",
        "accuracy": version_record["metrics"]["accuracy"],
        "precision_weighted": version_record["metrics"]["precision_macro"],
        "recall_weighted": version_record["metrics"]["recall_macro"],
        "f1_weighted": version_record["metrics"]["f1_weighted"],
        "f1_macro": version_record["metrics"]["f1_macro"],
        "quality_gate_passed": version_record["quality_gate_passed"],
        "promotion_decision": "accepted",
    }

    MODEL_REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    # The manifest records the version shown as current by dashboard and metadata
    # helpers, so it must move with the registry record.
    manifest = json.loads(VERSION_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["current_version"] = target_version
    VERSION_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"PASS: Successfully rolled back to version {target_version}")
    print(f"  Model path: {version_record['model_path']}")
    print(f"  Accuracy: {version_record['metrics']['accuracy']:.4f}")


def main() -> None:
    """Parse the requested target version and run the rollback update."""
    if len(sys.argv) < 2:
        candidates = get_rollback_candidates()
        print("Usage: python -m scripts.rollback_model <version>")
        print(f"Available versions: {', '.join(candidates)}")
        sys.exit(1)

    target_version = sys.argv[1]
    rollback_model(target_version)


if __name__ == "__main__":
    main()
