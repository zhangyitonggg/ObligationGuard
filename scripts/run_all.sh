#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${OBLIGATIONGUARD_RUN_ROOT:?Set OBLIGATIONGUARD_RUN_ROOT to the experiment output directory}"
: "${OBLIGATIONGUARD_CACHE_ROOT:?Set OBLIGATIONGUARD_CACHE_ROOT to a cache directory}"
export HF_HOME="${HF_HOME:-$OBLIGATIONGUARD_CACHE_ROOT/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$OBLIGATIONGUARD_CACHE_ROOT/torch}"
python -m obligationguard positive --paper configs/paper.toml --runtime "${1:-configs/runtime.toml}"
