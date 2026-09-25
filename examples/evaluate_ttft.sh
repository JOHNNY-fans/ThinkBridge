#!/usr/bin/env bash
# HF TTFT：每行独立记录首个实际回答 token，含输入准备、F/R 算 z 和 F prefill。
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_SIZE="${MODEL_SIZE:-0.6B}"
MODE="${MODE:-single-turn}"  # single-turn 或 multi-turn
WARMUP_SAMPLES="${WARMUP_SAMPLES:-2}" # 预热不计时
if [[ "$MODE" == multi-turn ]]; then
  GPU="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"  # 每卡独立 HF 副本
  if [[ "$MODEL_SIZE" == 4B ]]; then
    BATCH_SIZE="${MULTITURN_BATCH_SIZE:-8}"
  else
    BATCH_SIZE="${MULTITURN_BATCH_SIZE:-16}"
  fi
else
  GPU="${EVAL_GPU:-0}"
  BATCH_SIZE="${TTFT_BATCH_SIZE:-1}" # 单轮默认 1，也可设为 8、16 等
fi
source scripts/evaluation_runtime.sh
dataset="$TEST_DATASET"
workers=1
if [[ "$MODE" == multi-turn ]]; then
  dataset="${MULTITURN_DATASET:-data/benchmark_dataset/mathchat_follow_up_test.json}"
  IFS=',' read -r -a devices <<< "$GPU"
  workers="${#devices[@]}"
fi
# 单轮：每行首个回答 token 后停止，不算 ACC；batch>1 是批量负载下的 TTFT。
# 多轮：继续完整回答，同一输出算 ACC、形成历史；下一轮的历史与输入准备计入 TTFT。
# R 始终逐题计算。F batch 独立于 R；尾批按实际题数处理。
think-bridge benchmark "${common_args[@]}" \
  --backend hf --mode "$MODE" --measurement ttft --hf-workers "$workers" \
  --dataset "$dataset" --output-dir "$EVAL_RUN_DIR/$MODE-ttft" \
  --batch-size "$BATCH_SIZE" --reasoner-batch-size 1 \
  --warmup-samples "$WARMUP_SAMPLES"
