#!/usr/bin/env bash
# HF 评估：把需要运行的块设为 1。无需位置参数。
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_SIZE="${MODEL_SIZE:-0.6B}"          # 0.6B 或 4B
GPU="${EVAL_GPU:-0}"                     # 一张 GPU，F/R 共用
RUN_SINGLE_TURN="${RUN_SINGLE_TURN:-1}"   # 单轮答案 ACC：F + R
RUN_MULTITURN="${RUN_MULTITURN:-0}"       # 多轮 ACC + TTFT：同一完整输出
REASONER_BATCH_SIZE="${REASONER_BATCH_SIZE:-64}" # R 算 z 的 batch；长题可调小
ANSWER_BATCH_SIZE="${ANSWER_BATCH_SIZE:-16}"    # F 解码 batch，独立于 R
# 多轮：每卡一个 HF 副本；R 逐题，F 按所设 batch 解码。
MULTITURN_GPUS="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
if [[ "$MODEL_SIZE" == 4B ]]; then
  MULTITURN_BATCH_SIZE="${MULTITURN_BATCH_SIZE:-8}"
else
  MULTITURN_BATCH_SIZE="${MULTITURN_BATCH_SIZE:-16}"
fi
TEST_DATASET="${TEST_DATASET:-data/benchmark_dataset/gsm_test.json}"
MULTITURN_DATASET="${MULTITURN_DATASET:-data/benchmark_dataset/mathchat_follow_up_test.json}"
# 默认读取训练选出的 checkpoint；也可在这里设置 REASONER_CHECKPOINT。

source scripts/evaluation_runtime.sh

# 1. 单轮：按题计算答案准确率。
if [[ "$RUN_SINGLE_TURN" == 1 ]]; then
  think-bridge benchmark "${common_args[@]}" --backend hf --mode single-turn \
    --dataset "$TEST_DATASET" --output-dir "$EVAL_RUN_DIR/single-turn" \
    --reasoner-batch-size "$REASONER_BATCH_SIZE" --batch-size "$ANSWER_BATCH_SIZE"
fi

# 2. 多轮：完整问题与回答历史；每轮重新算 z，同一完整回答计算 ACC 和 TTFT。
if [[ "$RUN_MULTITURN" == 1 ]]; then
  IFS=',' read -r -a devices <<< "$MULTITURN_GPUS"
  CUDA_VISIBLE_DEVICES="$MULTITURN_GPUS" think-bridge benchmark "${common_args[@]}" --backend hf --mode multi-turn --measurement ttft \
    --dataset "$MULTITURN_DATASET" --output-dir "$EVAL_RUN_DIR/multi-turn" \
    --hf-workers "${#devices[@]}" --reasoner-batch-size 1 --batch-size "$MULTITURN_BATCH_SIZE"
fi
