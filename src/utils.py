"""Shared utilities for training and evaluation scripts."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Set random seed across major libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return config


def ensure_output_dir(path: str) -> Path:
    """Create an output directory if it does not exist."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_logger(output_dir: str | Path, name: str = "akkadian_mt") -> logging.Logger:
    """Create a simple stdout/file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = Path(output_dir) / "run.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def save_config(config: Dict[str, Any], output_dir: str | Path, filename: str = "config.yaml") -> Path:
    """Persist a YAML config snapshot."""
    output_path = Path(output_dir) / filename
    with output_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config, fp, allow_unicode=True, sort_keys=False)
    return output_path


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Write a JSON file with stable formatting."""
    with Path(path).open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def list_checkpoints(output_dir: str | Path) -> List[Path]:
    """List trainer checkpoints sorted by step."""
    base = Path(output_dir)
    checkpoints = [path for path in base.glob("checkpoint-*") if path.is_dir()]
    return sorted(checkpoints, key=lambda path: int(path.name.split("-")[-1]))


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
    return predictions
