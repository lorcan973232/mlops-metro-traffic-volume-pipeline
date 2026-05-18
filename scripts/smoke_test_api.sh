#!/usr/bin/env bash
set -euo pipefail

# The smoke test is the quickest proof that the deployed service is the real
# traffic-volume classifier. It checks `/health` first, then sends a valid
# prediction payload to the same `/predict` route used by the browser UI.
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
if payload.get("feature_count") != 15:
    raise SystemExit(f"Health response feature_count mismatch: {payload}")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

# The payload uses all 15 trained features. If any name or response field drifts,
# this script fails before the Docker or Kind evidence is accepted.
curl -fsS "${API_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":{"temp":288.28,"rain_1h":0.0,"snow_1h":0.0,"clouds_all":40.0,"hour":17,"month":10,"day_of_week":2,"is_weekend":0,"is_holiday":0,"weather_main":"Clouds","lag_1h_volume":5545.0,"lag_24h_volume":6015.0,"lag_168h_volume":5365.0,"rolling_3h_volume":5480.0,"rolling_24h_volume":4210.0}}' \
  -o "${PREDICT_RESPONSE}"
"${PYTHON_CMD}" - "${PREDICT_RESPONSE}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
prediction = payload.get("prediction")
if prediction not in {0, 1}:
    raise SystemExit(f"Invalid classification prediction response: {payload}")
if payload.get("prediction_label") not in {"normal traffic", "high traffic"}:
    raise SystemExit(f"Prediction label mismatch: {payload}")
if not payload.get("model_version"):
    raise SystemExit(f"Prediction response does not expose model_version: {payload}")
if payload.get("target") != "high_traffic":
    raise SystemExit(f"Prediction response target mismatch: {payload}")
if "confidence" not in payload:
    raise SystemExit(f"Prediction response does not expose confidence: {payload}")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
