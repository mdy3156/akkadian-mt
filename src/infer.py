"""Inference CLI for exported ByT5 checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .data import TASK_PREFIX
from .preprocess import postprocess_english_batch, preprocess_akkadian_text
from .utils import create_logger, ensure_output_dir


@torch.inference_mode()
def generate_predictions(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int,
    max_source_length: int,
    max_new_tokens: int,
    num_beams: int,
    device: torch.device,
) -> List[str]:
    """Run batched text generation."""
    predictions: List[str] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return postprocess_english_batch(predictions)


def parse_args() -> argparse.Namespace:
    """Parse inference CLI arguments."""
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned ByT5 checkpoint.")
    parser.add_argument("--model_path", required=True, help="Path to saved model/checkpoint directory.")
    parser.add_argument("--input_path", required=True, help="Path to input CSV.")
    parser.add_argument("--output_path", required=True, help="Path to output CSV.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")
    parser.add_argument("--max_source_length", type=int, default=512, help="Maximum input length.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum generated tokens.")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam size for generation.")
    return parser.parse_args()


def main() -> None:
    """Run checkpoint inference on a CSV file."""
    args = parse_args()
    output_path = Path(args.output_path)
    ensure_output_dir(output_path.parent)
    logger = create_logger(output_path.parent, name="akkadian_mt_infer")

    try:
        input_df = pd.read_csv(args.input_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input CSV not found: {args.input_path}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to read input CSV: {args.input_path}") from exc

    source_column = None
    for candidate in ("source", "transliteration"):
        if candidate in input_df.columns:
            source_column = candidate
            break
    if source_column is None:
        raise ValueError("Input CSV must contain either 'source' or 'transliteration' column.")

    texts = input_df[source_column].fillna("").map(preprocess_akkadian_text).tolist()
    prefixed_texts = [TASK_PREFIX + text for text in texts]

    logger.info("Loading model from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logger.info("Running generation for %d rows", len(prefixed_texts))
    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        texts=prefixed_texts,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        device=device,
    )

    output_df = pd.DataFrame({"prediction": predictions})
    if "id" in input_df.columns:
        output_df.insert(0, "id", input_df["id"])
    output_df.to_csv(output_path, index=False)
    logger.info("Saved predictions to %s", output_path)


if __name__ == "__main__":
    main()
