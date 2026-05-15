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
if payload.get("task_type") != "classification":
    raise SystemExit(f"Health response does not expose classification task type: {payload}")
if not payload.get("model_version"):
    raise SystemExit(f"Health response does not expose model_version: {payload}")
if payload.get("feature_count") != 11:
    raise SystemExit(f"Health response feature_count mismatch: {payload}")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

curl -fsS "${API_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":{"fixed_acidity":7.4,"volatile_acidity":0.7,"citric_acid":0.0,"residual_sugar":1.9,"chlorides":0.076,"free_sulfur_dioxide":11.0,"total_sulfur_dioxide":34.0,"density":0.9978,"ph":3.51,"sulphates":0.56,"alcohol":9.4}}' \
  -o "${PREDICT_RESPONSE}"
"${PYTHON_CMD}" - "${PREDICT_RESPONSE}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
prediction = payload.get("prediction")
if prediction not in {0, 1}:
    raise SystemExit(f"Invalid classification prediction response: {payload}")
if payload.get("prediction_label") not in {"standard quality", "good quality"}:
    raise SystemExit(f"Prediction label mismatch: {payload}")
if not payload.get("model_version"):
    raise SystemExit(f"Prediction response does not expose model_version: {payload}")
if payload.get("target") != "quality_label":
    raise SystemExit(f"Prediction response target mismatch: {payload}")
if "confidence" not in payload:
    raise SystemExit(f"Prediction response does not expose confidence: {payload}")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
