#!/usr/bin/env bash
set -euo pipefail

TRAIN_PATH="${1:-data/raw/kaggle/train.csv}"
VALID_PATH="${2:-}"
OUTPUT_DIR="${3:-outputs/byt5-base}"
CONFIG_PATH="${4:-configs/byt5_base.yaml}"
EXTRA_TRAIN_PATH="${5:-}"
MODEL_NAME_OVERRIDE="${6:-}"

if [[ -n "${VALID_PATH}" ]]; then
  CMD=(
    python -m src.train
    --config "${CONFIG_PATH}"
    --train_path "${TRAIN_PATH}"
    --valid_path "${VALID_PATH}"
    --output_dir "${OUTPUT_DIR}"
  )
else
  CMD=(
    python -m src.train
    --config "${CONFIG_PATH}"
    --train_path "${TRAIN_PATH}"
    --output_dir "${OUTPUT_DIR}"
  )
fi

if [[ -n "${EXTRA_TRAIN_PATH}" ]]; then
  CMD+=(--extra_train_paths "${EXTRA_TRAIN_PATH}")
fi

if [[ -n "${MODEL_NAME_OVERRIDE}" ]]; then
  CMD+=(--model_name "${MODEL_NAME_OVERRIDE}")
fi

"${CMD[@]}"
