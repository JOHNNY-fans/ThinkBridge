#!/usr/bin/env bash
# Train R; select the checkpoint by free-generation true-z accuracy.
set -euo pipefail
# Internal implementation, sourced by the two model-specific examples.
if (( $# != 0 )); then echo "Edit the training script settings; no positional arguments are needed." >&2; exit 2; fi
source scripts/runtime.sh
# The public entry supplies physical batch, GA, and separate GPU lists.
IFS=',' read -r -a reasoner_devices <<< "$TRAIN_GPUS"
STAGE1_WORLD=${#reasoner_devices[@]}
case "$STAGE1_GENERATION_BACKEND" in
  vllm) STAGE1_VISIBLE_DEVICES="$VLLM_GPUS,$TRAIN_GPUS" ;;
  torch) STAGE1_VISIBLE_DEVICES="$TRAIN_GPUS" ;;
  *) echo 'STAGE1_GENERATION_BACKEND must be vllm or torch' >&2; exit 2 ;;
esac
printf 'R backend=%s training_GPUs=%s service_GPUs=%s batch_per_gpu=%s GA=%s effective_batch=%s\n' \
  "$STAGE1_GENERATION_BACKEND" "$TRAIN_GPUS" "$VLLM_GPUS" "$BATCH_SIZE" \
  "$GRADIENT_ACCUMULATION_STEPS" "$((STAGE1_WORLD * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
common_args=(--seed "${SEED:-42}")
[[ "${NO_PROGRESS:-0}" != 1 ]] || common_args+=(--no_progress)
reasoner_extra=(--specificity_wrong_gradient live)
[[ -z "${REASONER_RESUME_CHECKPOINT:-}" ]] || reasoner_extra+=(--resume_from_checkpoint "$REASONER_RESUME_CHECKPOINT")
[[ -z "${SPECIFICITY_TEMPERATURE:-}" ]] || reasoner_extra+=(--specificity_temperature "$SPECIFICITY_TEMPERATURE")
[[ -z "${SPECIFICITY_NEGATIVE_KL_CAP:-}" ]] || reasoner_extra+=(--specificity_negative_kl_cap "$SPECIFICITY_NEGATIVE_KL_CAP")
if [[ -n "${RUN_LABEL:-}" && ! "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "Invalid RUN_LABEL" >&2; exit 2
fi
data_args=(--population staged --train_dataset "$TRAIN_DATASET"
  --train_behavior "$TRAIN_BEHAVIOR" --train_direct_answer_source "$TRAIN_DIRECT")
prepare_stage_invocation "$REASONER_PROJECT_DIR" reasoner-sft
CUDA_VISIBLE_DEVICES="$STAGE1_VISIBLE_DEVICES" NPROC_PER_NODE="$STAGE1_WORLD" MASTER_PORT=29592 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
think-bridge reasoner-sft \
  --model "$BASE_MODEL_DIR" \
  --tokenizer "$BASE_MODEL_DIR" \
  "${data_args[@]}" \
  --eval_dataset "$EVAL_DATASET" \
  --eval_behavior "$EVAL_BEHAVIOR" \
  --max_eval_samples "${MAX_EVAL_SAMPLES:-500}" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size "${REASONER_EVAL_BATCH_SIZE:-16}" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --physical_batch_size "$BATCH_SIZE" \
  --num_train_epochs "${REASONER_EPOCHS:-2}" \
  --max_steps "${REASONER_MAX_STEPS:--1}" \
  --learning_rate 0.0002 \
  --weight_decay 0.01 \
  --warmup_steps 40 \
  --max_grad_norm 1.0 \
  --eval_steps 50 \
  --save_steps 50 \
  --logging_steps 10 \
  --save_total_limit 10 \
  --gradient_checkpointing "${REASONER_GRADIENT_CHECKPOINTING:-false}" \
  --reasoner_layers "${REASONER_LAYERS:-2}" \
  --reasoner_loop_steps "${REASONER_LOOP_STEPS:-2}" \
  --latent_steps "${REASONER_LATENT_STEPS:-1}" \
  --latents_per_step "${REASONER_LATENTS_PER_STEP:-64}" \
  --ce_weight "${CE_WEIGHT:-1}" \
  --match_weight "${MATCH_WEIGHT:-1}" \
  --specific_weight "${SPECIFICITY_WEIGHT:-1}" \
  --specificity_include_direct false \
  --distillation_populations BC \
  --reasoner_dropout_p "${REASONER_DROPOUT_P:-0.1}" \
  --reasoner_dropout_views "${REASONER_DROPOUT_VIEWS:-2}" \
  --specificity_donors_per_owner "${SPECIFICITY_DONORS_PER_OWNER:-2}" \
  --generation_backend "${STAGE1_GENERATION_BACKEND:-vllm}" \
  --vllm_port "$SERVICE_PORT" \
  --vllm_physical_batch_size "${SERVING_PHYSICAL_BATCH_SIZE:-$PROFILE_SERVING_PHYSICAL}" \
  --vllm_max_in_flight 4 \
  --vllm_max_pending_microsteps "${ROUTE1_PREFETCH_MICROSTEPS:-$PROFILE_PREFETCH}" \
  --vllm_gpu_memory_utilization "$SERVICE_MEMORY" \
  --metric_for_best_model route1.true_z_full_accuracy \
  --metric_aggregation mean \
  --output_dir "$REASONER_PROJECT_DIR" \
  --run_locator "$RUN_LOCATOR" \
  "${common_args[@]}" \
  --generation_seed "${SEED:-42}" \
  ${reasoner_extra[@]+"${reasoner_extra[@]}"} \
  2>&1 | tee "$STAGE_LOG"

REASONER_RUN_DIR="$("$PYTHON_BIN" -m think_bridge.training.stage_run_locator_cli \
  --locator "$RUN_LOCATOR" --project-dir "$REASONER_PROJECT_DIR" --stage reasoner-sft --require-completed)"
printf 'Route1 complete: %s\n' "$REASONER_RUN_DIR"
