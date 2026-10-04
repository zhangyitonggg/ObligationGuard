#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OBLIGATIONGUARD_CACHE_ROOT:?Set OBLIGATIONGUARD_CACHE_ROOT to a cache directory}"
: "${OBLIGATIONGUARD_INFERENCE_ENV:?Set OBLIGATIONGUARD_INFERENCE_ENV to a separate virtual environment directory}"
python -m venv "$OBLIGATIONGUARD_INFERENCE_ENV"
"$OBLIGATIONGUARD_INFERENCE_ENV/bin/python" -m pip install --cache-dir "$OBLIGATIONGUARD_CACHE_ROOT/pip" -r requirements-inference.txt
echo "Set OBLIGATIONGUARD_INFERENCE_PYTHON=$OBLIGATIONGUARD_INFERENCE_ENV/bin/python"
