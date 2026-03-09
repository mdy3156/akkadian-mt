"""Evaluate a checkpoint on a deterministic subset of the canonical train data."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .data import TASK_PREFIX
from .metrics import compute_translation_metrics
from .preprocess import preprocess_akkadian_text, preprocess_english_text
from .utils import create_logger, ensure_output_dir, generate_predictions, load_yaml_config, save_json, set_seed

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a deterministic subset of data/train.csv.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    return parser.parse_args()


def load_eval_subset(input_path: str, sample_ratio: float, seed: int) -> pd.DataFrame:
    """Load and deterministically sample an evaluation subset."""
    try:
        dataframe = pd.read_csv(input_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input CSV not found: {input_path}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to read input CSV: {input_path}") from exc

    eval_df = dataframe.copy()
    required_columns = {"source", "target"}
    missing_columns = required_columns - set(eval_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input CSV must contain columns: {missing}")

    eval_df["id"] = [f"row-{index:06d}" for index in range(len(eval_df))]
    eval_df["raw_source"] = eval_df["source"].fillna("").astype(str)
    eval_df["raw_target"] = eval_df["target"].fillna("").astype(str)
    eval_df["source"] = eval_df["raw_source"].map(preprocess_akkadian_text)
    eval_df["target"] = eval_df["raw_target"].map(preprocess_english_text)
    eval_df = eval_df[(eval_df["source"] != "") & (eval_df["target"] != "")].reset_index(drop=True)

    if eval_df.empty:
        raise ValueError("No valid evaluation rows remained after preprocessing.")
    if not 0.0 < sample_ratio <= 1.0:
        raise ValueError("sample_ratio must be in the interval (0, 1].")

    if sample_ratio < 1.0:
        eval_df = eval_df.sample(frac=sample_ratio, random_state=seed).reset_index(drop=True)
    return eval_df



def main() -> None:
    """Run deterministic evaluation for a saved checkpoint."""
    args = parse_args()
    config = load_yaml_config(args.config)
    set_seed(int(config["seed"]))
    output_dir = ensure_output_dir(config["output_dir"])
    logger = create_logger(output_dir, name="akkadian_mt_eval")

    logger.info("Loading evaluation subset from %s", config["input_path"])
    eval_df = load_eval_subset(config["input_path"], float(config["sample_ratio"]), int(config["seed"]))
    logger.info("Evaluation rows: %d", len(eval_df))

    logger.info("Loading model from %s", config["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    prefixed_texts = [TASK_PREFIX + text for text in eval_df["source"].tolist()]
    logger.info("Running generation")
    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        texts=prefixed_texts,
        batch_size=int(config["batch_size"]),
        max_source_length=int(config["max_source_length"]),
        max_new_tokens=int(config["max_new_tokens"]),
        num_beams=int(config["num_beams"]),
        device=device,
    )

    metrics = compute_translation_metrics(predictions, eval_df["target"].tolist())
    source_char_lengths = np.array([len(text) for text in eval_df["source"].tolist()], dtype=np.float64)
    target_char_lengths = np.array([len(text) for text in eval_df["target"].tolist()], dtype=np.float64)
    prediction_char_lengths = np.array([len(text) for text in predictions], dtype=np.float64)
    source_word_lengths = np.array([len(text.split()) for text in eval_df["source"].tolist()], dtype=np.float64)
    target_word_lengths = np.array([len(text.split()) for text in eval_df["target"].tolist()], dtype=np.float64)
    prediction_word_lengths = np.array([len(text.split()) for text in predictions], dtype=np.float64)
    metrics.update(
        {
            "model_path": config["model_path"],
            "input_path": config["input_path"],
            "sample_ratio": float(config["sample_ratio"]),
            "seed": int(config["seed"]),
            "num_rows": len(eval_df),
            "avg_source_chars": round(float(source_char_lengths.mean()), 4),
            "avg_target_chars": round(float(target_char_lengths.mean()), 4),
            "avg_prediction_chars": round(float(prediction_char_lengths.mean()), 4),
            "avg_source_words": round(float(source_word_lengths.mean()), 4),
            "avg_target_words": round(float(target_word_lengths.mean()), 4),
            "avg_prediction_words": round(float(prediction_word_lengths.mean()), 4),
        }
    )
    save_json(metrics, output_dir / "eval_metrics.json")
    logger.info("Saved metrics to %s", output_dir / "eval_metrics.json")

    result_df = pd.DataFrame(
        {
            "id": eval_df["id"],
            "source_raw": eval_df["raw_source"],
            "source_preprocessed": eval_df["source"],
            "prediction": predictions,
            "target": eval_df["target"],
        }
    )
    result_df.to_csv(output_dir / "eval_predictions.csv", index=False)
    logger.info("Saved predictions to %s", output_dir / "eval_predictions.csv")


if __name__ == "__main__":
    main()
