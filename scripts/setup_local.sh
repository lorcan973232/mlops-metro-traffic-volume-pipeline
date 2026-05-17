#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env_paths.sh
source "${SCRIPT_DIR}/env_paths.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Creating local virtual environment in .venv"
"${PYTHON_BIN}" - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit("Python 3.11 or 3.12 is required for this pinned artefact environment.")
PY
"${PYTHON_BIN}" -m venv .venv

if [[ -f ".venv/Scripts/python.exe" ]]; then
  VENV_PYTHON=".venv/Scripts/python.exe"
else
  VENV_PYTHON=".venv/bin/python"
fi

"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install -r requirements.txt

"${VENV_PYTHON}" - <<'PY'
required = ["flask", "joblib", "numpy", "openpyxl", "pandas", "pytest", "sklearn", "yaml"]
missing = []
for package in required:
    try:
        __import__(package)
    except Exception:
        missing.append(package)
if missing:
    raise SystemExit("Missing imports after setup: " + ", ".join(missing))
print("PASS: dependency imports verified")
PY

cat <<'GUIDANCE'

Optional local tooling for full artefact verification:
- Python 3.11 or 3.12: https://www.python.org/downloads/
- Docker Desktop: https://docs.docker.com/desktop/
- Kind: https://kind.sigs.k8s.io/docs/user/quick-start/#installation
- kubectl: https://kubernetes.io/docs/tasks/tools/

After installation, run:
  scripts/check_setup.sh
  make full-local-verify

Activate the project environment before running README commands:
  source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

Or call Python directly:
  .venv/bin/python -m pytest -q
  .venv/Scripts/python.exe -m pytest -q
GUIDANCE

echo "PASS: local Bash setup completed."
