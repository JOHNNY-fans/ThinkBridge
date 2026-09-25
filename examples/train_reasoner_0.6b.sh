#!/usr/bin/env bash
# Train Qwen3-0.6B R for 2 epochs; select by validation answer accuracy.
# Batch = physical samples per training GPU per forward/backward pass.
# vLLM: 16 x 4 training GPUs x GA 2 = 128; 4 separate GPUs serve frozen F.
# Torch: 16 x 8 training GPUs x GA 1 = 128; no separate service GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_SIZE="0.6B"
BATCH_SIZE="${BATCH_SIZE:-16}"
STAGE1_GENERATION_BACKEND="${STAGE1_GENERATION_BACKEND:-vllm}"

# Choose visible GPUs and accumulation; only training GPUs count toward batch.
if [[ "$STAGE1_GENERATION_BACKEND" == torch ]]; then
  TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
  VLLM_GPUS=""
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
else
  TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
  VLLM_GPUS="${VLLM_GPUS:-0,1,2,3}"
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
fi

# Optional settings. Training-validation R batch remains 1.
# BASE_MODEL_DIR="models/Qwen3-0.6B"
# DATA_ROOT="data/stage0/0.6B"
# REASONER_EPOCHS=2

# Shared launcher: prepares paths, runs training, and saves the selected checkpoint.
source scripts/train_reasoner.sh
