"""Inspect or enforce the saved strict model quality gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluate import QUALITY_GATE_REPORT_PATH


def load_quality_gate(path: Path = QUALITY_GATE_REPORT_PATH) -> dict[str, Any]:
    """Load the quality-gate report written by `python -m src.evaluate`."""
    if not path.exists():
        raise FileNotFoundError(
            f"Quality-gate report not found at {path}. Run `python -m src.evaluate` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Print the saved gate and optionally fail when the candidate is rejected."""
    parser = argparse.ArgumentParser(description="Inspect the saved model quality gate.")
    parser.add_argument(
        "--fail-on-rejection",
        action="store_true",
        help="Exit non-zero when the saved quality gate rejected the model.",
    )
    args = parser.parse_args()
    gate = load_quality_gate()
    print(json.dumps(gate, indent=2, sort_keys=True))
    if args.fail_on_rejection and not gate.get("passed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
