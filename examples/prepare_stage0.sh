#!/usr/bin/env bash
# Prepare model-specific artifacts from the shared raw train/validation split.
set -euo pipefail
cd "$(dirname "$0")/.."
if (( $# > 1 )); then echo "Usage: bash $0 [0.6B|4B]" >&2; exit 2; fi
MODEL_SIZE="${1:-0.6B}"
source scripts/runtime.sh
args=(--model "$BASE_MODEL_DIR" --train "$TRAIN_SOURCE" \
      --validation "$VALIDATION_SOURCE" --output "$DATA_ROOT" \
      --backend "${STAGE0_BACKEND:-vllm}" --seed "${SEED:-42}")
[[ "${NO_PROGRESS:-0}" != 1 ]] || args+=(--no-progress)
CUDA_VISIBLE_DEVICES="${STAGE0_GPU:-0}" think-bridge prepare-stage0 "${args[@]}"
