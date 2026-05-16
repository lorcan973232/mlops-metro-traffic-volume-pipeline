#!/usr/bin/env bash
set -euo pipefail

python -m compileall app src tests scripts
python -m src.data
python -m src.preprocess
python -m src.model_selection
python -m src.train
python -m src.evaluate
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
