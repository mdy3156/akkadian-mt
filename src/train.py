"""CLI for fine-tuning ByT5 on Akkadian-English parallel data."""

from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from .data import TASK_PREFIX, build_hf_dataset, load_parallel_data, prepare_data_collator
from .metrics import build_compute_metrics, decode_prediction_batch
from .utils import create_logger, ensure_output_dir, list_checkpoints, load_yaml_config, save_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune ByT5 for Akkadian to English translation.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--train_path", required=True, help="Path to training CSV.")
    parser.add_argument("--valid_path", default=None, help="Optional path to validation CSV.")
    parser.add_argument("--output_dir", required=True, help="Directory to save checkpoints and outputs.")
    parser.add_argument("--model_name", default=None, help="Optional model name override.")
    parser.add_argument("--resume_from_checkpoint", default=None, help="Checkpoint path to resume from.")
    return parser.parse_args()


def build_training_arguments(config: Dict[str, Any], output_dir: str) -> Seq2SeqTrainingArguments:
    """Construct Seq2SeqTrainingArguments from config values."""
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "per_device_eval_batch_size": config["per_device_eval_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "learning_rate": float(config["learning_rate"]),
        "num_train_epochs": float(config["num_train_epochs"]),
        "warmup_ratio": float(config["warmup_ratio"]),
        "weight_decay": float(config["weight_decay"]),
        "logging_steps": int(config["logging_steps"]),
        "eval_steps": int(config["eval_steps"]),
        "save_steps": int(config["save_steps"]),
        "save_total_limit": int(config["save_total_limit"]),
        "predict_with_generate": bool(config["predict_with_generate"]),
        "generation_max_length": int(config["generation_max_length"]),
        "generation_num_beams": int(config["generation_num_beams"]),
        "fp16": bool(config["fp16"]),
        "bf16": bool(config["bf16"]),
        "seed": int(config["seed"]),
        "logging_strategy": config.get("logging_strategy", "steps"),
        "save_strategy": config.get("save_strategy", "steps"),
        "load_best_model_at_end": True,
        "metric_for_best_model": config.get("metric_for_best_model", "combined_score"),
        "greater_is_better": bool(config.get("greater_is_better", True)),
        "report_to": "none",
        "dataloader_pin_memory": True,
    }

    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = config.get("eval_strategy", "steps")
    elif "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = config.get("eval_strategy", "steps")

    return Seq2SeqTrainingArguments(**kwargs)


def resolve_train_valid_dataframes(
    train_df: pd.DataFrame,
    valid_df: Optional[pd.DataFrame],
    validation_split_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve explicit train/validation dataframes."""
    if valid_df is not None:
        return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)

    if not 0.0 < validation_split_ratio < 1.0:
        raise ValueError("validation_split_ratio must be between 0 and 1 when valid_path is not provided.")

    validation_df = train_df.sample(frac=validation_split_ratio, random_state=seed)
    training_df = train_df.drop(index=validation_df.index)
    return training_df.reset_index(drop=True), validation_df.reset_index(drop=True)


def save_validation_predictions(
    trainer: Seq2SeqTrainer,
    dataset,
    tokenizer,
    raw_valid_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate and save validation predictions to CSV."""
    prediction_output = trainer.predict(dataset, metric_key_prefix="predict")
    decoded_predictions = decode_prediction_batch(prediction_output.predictions, tokenizer)

    result_df = raw_valid_df.copy()
    result_df["prefixed_source"] = result_df["source"].map(lambda value: TASK_PREFIX + value)
    result_df["prediction"] = decoded_predictions
    result_df.to_csv(output_path, index=False)


def export_best_checkpoint(trainer: Seq2SeqTrainer, tokenizer, output_dir: Path, logger) -> Optional[Path]:
    """Copy the best checkpoint into a stable directory for downstream inference."""
    best_checkpoint = trainer.state.best_model_checkpoint
    if not best_checkpoint:
        logger.warning("Best checkpoint was not recorded; skipping best_checkpoint export.")
        return None

    destination = output_dir / "best_checkpoint"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(best_checkpoint, destination)
    tokenizer.save_pretrained(destination)
    logger.info("Exported best checkpoint to %s", destination)
    return destination


def main() -> None:
    """Run fine-tuning end to end."""
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.model_name:
        config["model_name"] = args.model_name

    output_dir = ensure_output_dir(args.output_dir)
    logger = create_logger(output_dir)
    save_config(config, output_dir)
    set_seed(int(config["seed"]))

    logger.info("Loading tokenizer and model: %s", config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])

    logger.info("Loading data")
    train_df = load_parallel_data(args.train_path)
    valid_df = load_parallel_data(args.valid_path) if args.valid_path else None
    train_df, raw_valid_df = resolve_train_valid_dataframes(
        train_df=train_df,
        valid_df=valid_df,
        validation_split_ratio=float(config.get("validation_split_ratio", 0.05)),
        seed=int(config["seed"]),
    )

    dataset_dict = build_hf_dataset(
        train_df=train_df,
        valid_df=raw_valid_df,
        tokenizer=tokenizer,
        max_source_length=int(config["max_source_length"]),
        max_target_length=int(config["max_target_length"]),
        seed=int(config["seed"]),
        preprocessing_num_workers=config.get("preprocessing_num_workers"),
    )

    training_args = build_training_arguments(config, str(output_dir))
    data_collator = prepare_data_collator(tokenizer=tokenizer, model=model)
    compute_metrics = build_compute_metrics(tokenizer)

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset_dict["train"],
        "eval_dataset": dataset_dict["validation"],
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Seq2SeqTrainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Seq2SeqTrainer(
        **trainer_kwargs,
    )

    logger.info("Starting training")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    logger.info("Evaluating best model")
    eval_metrics = trainer.evaluate(metric_key_prefix="eval")
    metrics: Dict[str, Any] = {**train_result.metrics, **eval_metrics}
    save_json(metrics, output_dir / "all_results.json")

    prediction_csv_path = output_dir / "validation_predictions.csv"
    save_validation_predictions(
        trainer=trainer,
        dataset=dataset_dict["validation"],
        tokenizer=tokenizer,
        raw_valid_df=raw_valid_df,
        output_path=prediction_csv_path,
    )

    export_best_checkpoint(trainer, tokenizer, output_dir, logger)

    checkpoints = [str(path) for path in list_checkpoints(output_dir)]
    save_json({"checkpoints": checkpoints}, output_dir / "checkpoints.json")
    logger.info("Training complete")


if __name__ == "__main__":
    main()
