#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-outputs/byt5-base/best_checkpoint}"
INPUT_PATH="${2:-data/test.csv}"
OUTPUT_PATH="${3:-outputs/predictions.csv}"
BATCH_SIZE="${4:-8}"

python -m src.infer \
  --model_path "${MODEL_PATH}" \
  --input_path "${INPUT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --batch_size "${BATCH_SIZE}"
