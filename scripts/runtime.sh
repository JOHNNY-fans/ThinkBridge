#!/usr/bin/env bash
# Shared launch settings. Use the Python environment selected by the caller.
set -euo pipefail
export THINK_BRIDGE_PROJECT_ROOT="$PWD"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONDONTWRITEBYTECODE=1
function think-bridge() { "$PYTHON_BIN" -m think_bridge.cli.main "$@"; }
case "$MODEL_SIZE" in
  0.6B) FAMILY=qwen3-0.6b; SERVICE_PORT=29601; SERVICE_MEMORY=0.8 ;;
  4B) FAMILY=qwen3-4b; SERVICE_PORT=29611; SERVICE_MEMORY=0.75 ;;
  *) echo 'MODEL_SIZE must be 0.6B or 4B' >&2; exit 2 ;;
esac
BASE_MODEL_DIR="${BASE_MODEL_DIR:-Qwen/Qwen3-$MODEL_SIZE}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-data/raw}"
TRAIN_SOURCE="${TRAIN_SOURCE:-$RAW_DATA_ROOT/train.json}"
VALIDATION_SOURCE="${VALIDATION_SOURCE:-$RAW_DATA_ROOT/validation.json}"
BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-data/benchmark_dataset}"
TEST_DATASET="${TEST_DATASET:-$BENCHMARK_DATA_DIR/gsm_test.json}"
DATA_ROOT="${DATA_ROOT:-data/stage0/$MODEL_SIZE}"
TRAIN_DATASET="${TRAIN_DATASET:-$DATA_ROOT/native.json}"
TRAIN_BEHAVIOR="${TRAIN_BEHAVIOR:-$DATA_ROOT/behavior.json}"
TRAIN_DIRECT="${TRAIN_DIRECT:-$DATA_ROOT/direct.json}"
EVAL_DATASET="${EVAL_DATASET:-$DATA_ROOT/validation.json}"
EVAL_BEHAVIOR="${EVAL_BEHAVIOR:-$DATA_ROOT/validation_behavior.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
REASONER_PROJECT_DIR="${REASONER_PROJECT_DIR:-$OUTPUT_ROOT/$FAMILY/${RUN_LABEL:-reasoner-sft}}"
EVALUATION_ROOT="${EVALUATION_ROOT:-$OUTPUT_ROOT/$FAMILY/evaluation}"
read -r PROFILE_SERVING_PHYSICAL PROFILE_PREFETCH < <(
  "$PYTHON_BIN" - "$FAMILY" <<'CONFIG'
import json,sys
from pathlib import Path
c=json.loads((Path('configs') / ('stage1_'+sys.argv[1].replace('qwen3-', 'qwen3_')+'.yaml')).read_text())
print(*(c[k] for k in ('route1_vllm_physical_chunk_size','route1_vllm_max_pending_microsteps')))
CONFIG
)
source scripts/stage_logging.sh

# Resolve the saved selection, never guess a run number or training step.
resolve_checkpoint_source() {
  local source="$1" project="$2" stage="$3" route="$4" pointer
  if [[ -z "$source" ]]; then
    if [[ ! -f "$project/latest-run.txt" ]]; then
      echo "No completed $stage run in $project; train it first or supply a checkpoint path." >&2
      return 2
    fi
    source="$("$PYTHON_BIN" -m think_bridge.training.stage_run_locator_cli \
      --locator "$project/latest-run.txt" --project-dir "$project" \
      --stage "$stage" --require-completed)" || return
  fi
  case "$route" in
    route1) pointer=best_reasoner_checkpoint.txt ;;
  esac
  if [[ -f "$source/$pointer" ]]; then
    "$PYTHON_BIN" -m think_bridge.training.checkpoint_pointer_cli \
      --run-dir "$source" --pointer "$source/$pointer" --expected-route "$route"
  else
    printf '%s\n' "$source"
  fi
}

