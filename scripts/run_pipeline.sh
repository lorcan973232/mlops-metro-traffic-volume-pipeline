#!/usr/bin/env bash
set -euo pipefail

# This is the Bash one-command verification path for a local machine or Git Bash.
# It runs the same ordered stages used in the README: compile, data,
# preprocessing, model selection, training, evaluation, registry metadata,
# prediction, Tier 3 evidence, monitoring, tests, lint, and Flask import. Docker
# and Kind are separate because they depend on local container tooling.
python -m compileall app src tests scripts
python -m src.data
python -m src.preprocess
python -m src.model_selection
python -m src.train
python -m src.evaluate --fail-on-rejection
python -m src.model_registry
python -m src.predict
python scripts/explain_model.py
python scripts/fairness_audit.py
python scripts/cost_benefit_analysis.py
python scripts/monitor.py
python scripts/check_drift.py
pytest -q
ruff check src tests scripts
python -c "from app.main import app; print('Flask import OK')"
