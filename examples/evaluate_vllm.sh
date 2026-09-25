#!/usr/bin/env bash
# vLLM 单轮答案 ACC。默认只有一张可见 GPU，DP=1、TP=1。
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_SIZE="${MODEL_SIZE:-0.6B}"
GPU="${EVAL_GPU:-0}"
ANSWER_BATCH_SIZE="${ANSWER_BATCH_SIZE:-8}"
REASONER_BATCH_SIZE="${REASONER_BATCH_SIZE:-64}"
# 0.50 是整张卡的比例，不是剩余显存比例。
# 同卡还要放 HF 的冻结 F、R 和算 z 的临时激活；此处存在两份 F 权重。
# 长输入或显存不足时，同时调小 R batch / 答案 batch / vLLM 比例。
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.50}"
source scripts/evaluation_runtime.sh
# eager 不预留 CUDA graph 显存；z 以 embedding 输入 vLLM，保留相同 think 边界。
think-bridge benchmark "${common_args[@]}" --backend vllm --mode single-turn \
  --dataset "$TEST_DATASET" --output-dir "$EVAL_RUN_DIR/single-turn-vllm" \
  --batch-size "$ANSWER_BATCH_SIZE" --reasoner-batch-size "$REASONER_BATCH_SIZE" \
  --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" --vllm-enforce-eager
