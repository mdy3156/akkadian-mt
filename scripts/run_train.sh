#!/usr/bin/env bash
set -euo pipefail

TRAIN_PATH="${1:-data/raw/kaggle/train.csv}"
VALID_PATH="${2:-}"
OUTPUT_DIR="${3:-outputs/byt5-base}"
CONFIG_PATH="${4:-configs/byt5_base.yaml}"

if [[ -n "${VALID_PATH}" ]]; then
  uv run python -m src.train \
    --config "${CONFIG_PATH}" \
    --train_path "${TRAIN_PATH}" \
    --valid_path "${VALID_PATH}" \
    --output_dir "${OUTPUT_DIR}"
else
  uv run python -m src.train \
    --config "${CONFIG_PATH}" \
    --train_path "${TRAIN_PATH}" \
    --output_dir "${OUTPUT_DIR}"
fi
