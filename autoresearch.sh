#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${PYTHON:-python}"
if [[ -x "$PWD/.venv/bin/python" ]]; then
    python_bin="$PWD/.venv/bin/python"
fi
exec "$python_bin" -u bench/autoresearch_memory_benchmark.py
