#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env_paths.sh
source "${SCRIPT_DIR}/env_paths.sh"

API_URL="${1:-http://127.0.0.1:8080}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="${PYTHON_BIN}"
elif [[ -f ".venv/Scripts/python.exe" ]]; then
  PYTHON_CMD=".venv/Scripts/python.exe"
elif [[ -f ".venv/bin/python" ]]; then
  PYTHON_CMD=".venv/bin/python"
else
  PYTHON_CMD="python"
fi
HEALTH_RESPONSE="$(mktemp)"
PREDICT_RESPONSE="$(mktemp)"
cleanup() {
  rm -f "${HEALTH_RESPONSE}" "${PREDICT_RESPONSE}"
}
trap cleanup EXIT

curl -fsS "${API_URL}/health" -o "${HEALTH_RESPONSE}"
"${PYTHON_CMD}" - "${HEALTH_RESPONSE}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "healthy" or payload.get("model_loaded") is not True:
    raise SystemExit(f"Invalid health response: {payload}")
if not payload.get("model_version"):
    raise SystemExit(f"Health response does not expose model_version: {payload}")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

curl -fsS "${API_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":{"fixed_acidity":7.0,"volatile_acidity":0.27,"citric_acid":0.36,"residual_sugar":20.7,"chlorides":0.045,"free_sulfur_dioxide":45.0,"total_sulfur_dioxide":170.0,"density":1.001,"pH":3.0,"sulphates":0.45,"alcohol":8.8}}' \
  -o "${PREDICT_RESPONSE}"
"${PYTHON_CMD}" - "${PREDICT_RESPONSE}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
prediction = payload.get("prediction")
if prediction not in {"low", "medium", "high"}:
    raise SystemExit(f"Invalid prediction response: {payload}")
if not payload.get("model_version"):
    raise SystemExit(f"Prediction response does not expose model_version: {payload}")
probabilities = payload.get("probabilities", {})
missing = {"low", "medium", "high"} - set(probabilities)
if missing:
    raise SystemExit(f"Prediction response missing probabilities for: {sorted(missing)}")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
