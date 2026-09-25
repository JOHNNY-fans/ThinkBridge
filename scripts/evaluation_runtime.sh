#!/usr/bin/env bash
# Shared paths/checkpoint selection; edit experiment settings in examples/.
source scripts/runtime.sh
export CUDA_VISIBLE_DEVICES="$GPU"
EVAL_RUN_DIR="${EVAL_OUTPUT_DIR:-$EVALUATION_ROOT/$(date +%Y%m%d-%H%M%S)-$$}"
reasoner_source="${REASONER_CHECKPOINT:-}"
reasoner_source="$(resolve_checkpoint_source "$reasoner_source" "$REASONER_PROJECT_DIR" reasoner-sft route1)"
common_args=(--model "$BASE_MODEL_DIR" --reasoner-checkpoint "$reasoner_source"
             --seed "${SEED:-42}")
[[ "${NO_PROGRESS:-0}" != 1 ]] || common_args+=(--no-progress)
printf 'R: %s\nOutput: %s\n' "$reasoner_source" "$EVAL_RUN_DIR"
