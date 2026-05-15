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
if payload.get("task_type") != "regression":
    raise SystemExit(f"Health response does not expose regression task type: {payload}")
if not payload.get("model_version"):
    raise SystemExit(f"Health response does not expose model_version: {payload}")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

curl -fsS "${API_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":{"relative_compactness":0.76,"surface_area":661.5,"wall_area":416.5,"roof_area":122.5,"overall_height":7.0,"orientation":2,"glazing_area":0.4,"glazing_area_distribution":5}}' \
  -o "${PREDICT_RESPONSE}"
"${PYTHON_CMD}" - "${PREDICT_RESPONSE}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
prediction = payload.get("prediction")
if not isinstance(prediction, (int, float)):
    raise SystemExit(f"Invalid regression prediction response: {payload}")
if prediction <= 0:
    raise SystemExit(f"Heating load prediction must be positive: {payload}")
if not payload.get("model_version"):
    raise SystemExit(f"Prediction response does not expose model_version: {payload}")
if payload.get("target") != "heating_load":
    raise SystemExit(f"Prediction response target mismatch: {payload}")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
