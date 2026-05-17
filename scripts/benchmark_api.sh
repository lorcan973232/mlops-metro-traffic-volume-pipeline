#!/bin/bash
set -e

# Bash wrapper for API latency evidence. Use it after the Flask, Docker, or Kind
# API is reachable; it delegates to the Python benchmark so Windows and Bash
# routes produce the same report format under reports/benchmarks/.

if [ -z "$1" ]; then
    echo "Usage: $0 <API_URL> [--samples N] [--warmup N]"
    echo "Example: $0 http://127.0.0.1:5000"
    exit 1
fi

API_URL="$1"
shift

mkdir -p reports/benchmarks

python scripts/benchmark_api.py "$API_URL" "$@"
