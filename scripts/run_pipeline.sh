#!/usr/bin/env bash
set -euo pipefail

python -m src.data
python -m src.preprocess
python -m src.train
python -m src.evaluate
python -m src.model_registry
python scripts/monitor.py
python scripts/check_drift.py

