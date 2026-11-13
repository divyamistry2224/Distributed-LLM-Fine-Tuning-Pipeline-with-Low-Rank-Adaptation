#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
accelerate launch --config_file "${ACCELERATE_CONFIG:-configs/accelerate_single.yaml}" -m src.train \
  --base_model "${BASE_MODEL:-Qwen/Qwen2.5-0.5B}" \
  --train_file "${TRAIN_FILE:-data/processed/train.jsonl}" \
  --eval_file "${EVAL_FILE:-data/processed/validation.jsonl}" \
  --output_dir "${OUTPUT_DIR:-runs/document-lora}" \
  --lora_config "${LORA_CONFIG:-configs/lora_config.yaml}" \
  --num_train_epochs "${EPOCHS:-3}" \
  --per_device_train_batch_size "${BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --learning_rate "${LR:-2e-4}" --gradient_checkpointing "$@"
