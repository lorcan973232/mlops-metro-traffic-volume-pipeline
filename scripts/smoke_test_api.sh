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
  -d '{"features":{"mean_radius":17.99,"mean_texture":10.38,"mean_perimeter":122.8,"mean_area":1001.0,"mean_smoothness":0.1184,"mean_compactness":0.2776,"mean_concavity":0.3001,"mean_concave_points":0.1471,"mean_symmetry":0.2419,"mean_fractal_dimension":0.07871,"radius_error":1.095,"texture_error":0.9053,"perimeter_error":8.589,"area_error":153.4,"smoothness_error":0.006399,"compactness_error":0.04904,"concavity_error":0.05373,"concave_points_error":0.01587,"symmetry_error":0.03003,"fractal_dimension_error":0.006193,"worst_radius":25.38,"worst_texture":17.33,"worst_perimeter":184.6,"worst_area":2019.0,"worst_smoothness":0.1622,"worst_compactness":0.6656,"worst_concavity":0.7119,"worst_concave_points":0.2654,"worst_symmetry":0.4601,"worst_fractal_dimension":0.1189}}' \
  -o "${PREDICT_RESPONSE}"
"${PYTHON_CMD}" - "${PREDICT_RESPONSE}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
prediction = payload.get("prediction")
if prediction not in {"malignant", "benign"}:
    raise SystemExit(f"Invalid prediction response: {payload}")
if not payload.get("model_version"):
    raise SystemExit(f"Prediction response does not expose model_version: {payload}")
probabilities = payload.get("probabilities", {})
missing = {"malignant", "benign"} - set(probabilities)
if missing:
    raise SystemExit(f"Prediction response missing probabilities for: {sorted(missing)}")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
