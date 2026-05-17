#!/usr/bin/env bash
set -euo pipefail

# This check is used locally and in Ubuntu GitHub Actions. It separates real
# project failures from missing setup tools such as Docker, Kind, kubectl, or an
# uncreated virtual environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env_paths.sh
source "${SCRIPT_DIR}/env_paths.sh"

MODE="${1:-full}"
REQUIRE_GH="${2:-}"
blocked=0
failed=0

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="${PYTHON_BIN}"
elif [[ -f ".venv/Scripts/python.exe" ]]; then
  PYTHON_CMD=".venv/Scripts/python.exe"
elif [[ -f ".venv/bin/python" ]]; then
  PYTHON_CMD=".venv/bin/python"
else
  PYTHON_CMD="python"
fi

print_check() {
  local status="$1"
  local name="$2"
  local detail="$3"
  echo "${status}: ${name} - ${detail}"
}

block() {
  print_check "BLOCKED_BY_LOCAL_SETUP" "$1" "$2"
  blocked=1
}

fail() {
  print_check "FAIL" "$1" "$2"
  failed=1
}

pass() {
  print_check "PASS" "$1" "$2"
}

require_command() {
  local name="$1"
  local guidance="$2"
  if command -v "${name}" >/dev/null 2>&1; then
    pass "${name}" "$(command -v "${name}")"
  else
    block "${name}" "${guidance}"
  fi
}

if [[ ! -f "README.md" || ! -f "requirements.txt" || ! -d "src" || ! -d "app" ]]; then
  fail "repository root" "Run this script from the repository root."
else
  pass "repository root" "$(pwd)"
fi

case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*)
    pass "windows shell guidance" "Git Bash detected. PowerShell scripts remain the recommended Windows path."
    ;;
  Linux*)
    pass "shell guidance" "Linux Bash detected. On Windows, do not use WSL bash unless WSL is configured."
    ;;
  *)
    pass "shell guidance" "Use PowerShell on Windows, Git Bash explicitly on Windows, or Bash on Linux/macOS."
    ;;
esac

require_command "${PYTHON_CMD}" "Install Python 3.11 or 3.12: https://www.python.org/downloads/"
require_command git "Install Git: https://git-scm.com/downloads"

if [[ "${REQUIRE_GH}" == "--require-gh" ]]; then
  require_command gh "Install GitHub CLI: https://cli.github.com/ then run: gh auth login"
fi

if command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
  if "${PYTHON_CMD}" - <<'PY'
import sys
import venv

if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(1)
PY
  then
    pass "python version and venv" "$("${PYTHON_CMD}" --version 2>&1)"
  else
    fail "python version and venv" "Use Python 3.11 or 3.12 with venv support."
  fi

  if "${PYTHON_CMD}" -m pip --version >/dev/null 2>&1; then
    pass "pip" "$("${PYTHON_CMD}" -m pip --version 2>&1)"
  else
    block "pip" "Install pip with Python or run: ${PYTHON_CMD} -m ensurepip --upgrade"
  fi

  if [[ -d ".venv" ]]; then
    pass "virtual environment" ".venv exists; selected interpreter is ${PYTHON_CMD}"
  else
    block "virtual environment" "Run: scripts/setup_local.sh"
  fi

  if "${PYTHON_CMD}" - <<'PY'
required = ["flask", "joblib", "numpy", "openpyxl", "pandas", "pytest", "sklearn", "yaml"]
missing = []
for package in required:
    try:
        __import__(package)
    except Exception:
        missing.append(package)
if missing:
    raise SystemExit(", ".join(missing))
PY
  then
    pass "python dependencies" "required packages import successfully"
  else
    block "python dependencies" "Run: ${PYTHON_CMD} -m pip install -r requirements.txt"
  fi
fi

if [[ "${MODE}" == "--python-only" ]]; then
  pass "deployment tooling" "Docker, Kind, and kubectl checks skipped for non-deployment CI."
else
  require_command docker "Install Docker Desktop: https://docs.docker.com/desktop/"
  require_command kind "Install Kind: https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
  require_command kubectl "Install kubectl: https://kubernetes.io/docs/tasks/tools/"

  if command -v docker >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then
      pass "docker daemon" "Docker daemon is running"
    else
      block "docker daemon" "Start Docker Desktop, then rerun scripts/check_setup.sh"
    fi
  fi
fi

if [[ "${failed}" -ne 0 ]]; then
  echo "FAIL: setup check failed. Fix failed checks above."
  exit 1
fi

if [[ "${blocked}" -ne 0 ]]; then
  echo "BLOCKED_BY_LOCAL_SETUP: install or start the blocked dependencies above."
  exit 1
fi

echo "PASS: local setup check passed."
