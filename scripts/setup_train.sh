#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OBLIGATIONGUARD_CACHE_ROOT:?Set OBLIGATIONGUARD_CACHE_ROOT to a cache directory}"
export HF_HOME="${HF_HOME:-$OBLIGATIONGUARD_CACHE_ROOT/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$OBLIGATIONGUARD_CACHE_ROOT/torch}"
python -m pip install --cache-dir "$OBLIGATIONGUARD_CACHE_ROOT/pip" -r requirements-train.txt
python -m pip install --cache-dir "$OBLIGATIONGUARD_CACHE_ROOT/pip" flash-attn==2.8.3 --no-build-isolation
python -m pip install --cache-dir "$OBLIGATIONGUARD_CACHE_ROOT/pip" -e . --no-deps
