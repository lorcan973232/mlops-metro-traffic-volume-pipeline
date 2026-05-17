from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VERSION_MANIFEST_PATH = Path("reports/model_registry/version_manifest.json")


# ==============================================================================
# Model version manifest
# ==============================================================================
#
# The project does not claim to use an external model registry. This small
# manifest records which evaluated model version is active, which versions passed
# the quality gate, and which versions could be used as rollback candidates.


def initialize_version_manifest() -> dict[str, Any]:
    """Create or load the repository-local model version manifest."""
    if VERSION_MANIFEST_PATH.exists():
        return json.loads(VERSION_MANIFEST_PATH.read_text(encoding="utf-8"))

    manifest = {
        "current_version": "1.0.0",
        "versions": [],
        "rollback_candidates": [],
    }
    VERSION_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERSION_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_version(version_str: str) -> tuple[int, int, int]:
    """Parse `major.minor.patch` so version increments are explicit."""
    parts = version_str.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid semantic version: {version_str}")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise ValueError(f"Invalid semantic version: {version_str}") from exc


def increment_patch_version(current_version: str) -> str:
    """Bump the patch number for a retrained model with the same design."""
    major, minor, patch = parse_version(current_version)
    return f"{major}.{minor}.{patch + 1}"


def increment_minor_version(current_version: str) -> str:
    """Bump the minor number when model design changes are intentionally larger."""
    major, minor, patch = parse_version(current_version)
    return f"{major}.{minor + 1}.0"


def increment_major_version(current_version: str) -> str:
    """Bump the major number for a breaking model-management change."""
    major, minor, patch = parse_version(current_version)
    return f"{major + 1}.0.0"


def get_next_version(
    current_version: str, increment_type: str = "patch"
) -> str:
    """Return the next semantic version requested by the retraining stage."""
    if increment_type == "patch":
        return increment_patch_version(current_version)
    elif increment_type == "minor":
        return increment_minor_version(current_version)
    elif increment_type == "major":
        return increment_major_version(current_version)
    else:
        raise ValueError(f"Invalid increment type: {increment_type}")


def get_current_version() -> str:
    """Read the active model version used when new metadata is registered."""
    manifest = initialize_version_manifest()
    return manifest.get("current_version", "1.0.0")


def register_version(
    version: str,
    model_path: Path,
    metadata_path: Path,
    metrics: dict[str, Any],
    data_hash: str,
    training_timestamp: str,
    quality_gate_passed: bool,
) -> None:
    """Add a model version and promote it only if the quality gate passed."""
    manifest = initialize_version_manifest()

    version_record = {
        "version": version,
        "timestamp": training_timestamp,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "f1_macro": metrics.get("f1_macro"),
            "f1_weighted": metrics.get("f1_weighted"),
            "precision_macro": metrics.get("precision_macro"),
            "recall_macro": metrics.get("recall_macro"),
            "roc_auc": metrics.get("roc_auc"),
        },
        "quality_gate_passed": quality_gate_passed,
        "status": "active" if quality_gate_passed else "rejected",
        "data_hash": data_hash,
        "training_timestamp": training_timestamp,
    }

    existing_versions = [
        record for record in manifest["versions"] if record.get("version") != version
    ]
    existing_versions.append(version_record)
    manifest["versions"] = existing_versions
    if quality_gate_passed:
        manifest["current_version"] = version
        manifest["rollback_candidates"] = [
            v["version"] for v in manifest["versions"] if v["quality_gate_passed"]
        ]

    VERSION_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def get_version_record(version: str) -> dict[str, Any] | None:
    """Return one version record for dashboard or rollback inspection."""
    manifest = initialize_version_manifest()
    for v in manifest.get("versions", []):
        if v["version"] == version:
            return v
    return None


def get_rollback_candidates() -> list[str]:
    """List previously accepted versions that are safe rollback candidates."""
    manifest = initialize_version_manifest()
    return manifest.get("rollback_candidates", [])


def main() -> None:
    manifest = initialize_version_manifest()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
