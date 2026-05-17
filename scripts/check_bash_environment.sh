#!/usr/bin/env bash
set -euo pipefail

echo "PASS: bash executable path: ${BASH:-unknown}"
echo "PASS: bash version: ${BASH_VERSION:-unknown}"
echo "PASS: uname: $(uname -a 2>/dev/null || true)"

case "$(uname -s 2>/dev/null || echo unknown)" in
  Linux*)
    echo "PASS: Linux/Ubuntu Bash route is available."
    ;;
  MINGW*|MSYS*|CYGWIN*)
    echo "PASS: Git Bash route is available on Windows."
    ;;
  *)
    echo "WARN: Bash is available, but the platform is not a standard Linux or Git Bash route."
    ;;
esac

for tool in docker kind kubectl; do
  if command -v "${tool}" >/dev/null 2>&1; then
    echo "PASS: ${tool} found at $(command -v "${tool}")"
  else
    echo "BLOCKED_BY_LOCAL_SETUP: ${tool} not found on PATH."
  fi
done

echo "PASS: .sh scripts can run in this shell."
