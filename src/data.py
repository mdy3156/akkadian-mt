"""Dataset loading and tokenization utilities."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

from .preprocess import preprocess_akkadian_text, preprocess_english_text

TASK_PREFIX = "translate Akkadian to English: "
REVERSE_TASK_PREFIX = "translate English to Akkadian: "


def load_parallel_data(path: str) -> pd.DataFrame:
    """Load a canonical parallel CSV file with source/target columns."""
    try:
        dataframe = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Parallel data file not found: {path}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file: {path}") from exc

    dataframe = dataframe.copy()
    # Compatibility shim for notebook-exported sentence data with a typoed column name.
    if "cssource" in dataframe.columns and "source" not in dataframe.columns:
        dataframe = dataframe.rename(columns={"cssource": "source"})

    required_columns = {"source", "target"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Parallel CSV must contain columns: {missing}")

    dataframe["raw_source"] = dataframe["source"].fillna("").astype(str)
    dataframe["raw_target"] = dataframe["target"].fillna("").astype(str)
    dataframe["source"] = dataframe["source"].fillna("").map(preprocess_akkadian_text)
    dataframe["target"] = dataframe["target"].fillna("").map(preprocess_english_text)

    dataframe = dataframe[(dataframe["source"] != "") & (dataframe["target"] != "")].reset_index(drop=True)
    if dataframe.empty:
        raise ValueError(f"No valid rows found in parallel data: {path}")
    return dataframe


def create_bidirectional_train_data(train_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Duplicate the training data with an English-to-Akkadian reverse task."""
    forward_df = train_df.copy()
    forward_df["input_text"] = forward_df["source"].map(lambda text: TASK_PREFIX + text)
    forward_df["target_text"] = forward_df["target"]

    backward_df = train_df.copy()
    backward_df["input_text"] = backward_df["target"].map(lambda text: REVERSE_TASK_PREFIX + text)
    backward_df["target_text"] = backward_df["source"]

    combined_df = pd.concat([forward_df, backward_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return combined_df


def tokenize_function(
    examples: Dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    max_source_length: int,
    max_target_length: int,
) -> Dict[str, Any]:
    """Tokenize a batch of translation examples for ByT5."""
    if "input_text" in examples and "target_text" in examples:
        inputs = [str(text) for text in examples["input_text"]]
        targets = [str(text) for text in examples["target_text"]]
    else:
        inputs = [TASK_PREFIX + text for text in examples["source"]]
        targets = [str(text) for text in examples["target"]]

    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        truncation=True,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_hf_dataset(
    train_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_source_length: int,
    max_target_length: int,
    valid_df: Optional[pd.DataFrame] = None,
    validation_split_ratio: float = 0.05,
    seed: int = 42,
    preprocessing_num_workers: Optional[int] = None,
    bidirectional_augmentation: bool = False,
) -> DatasetDict:
    """Build tokenized Hugging Face datasets for train/validation."""
    if bidirectional_augmentation:
        train_df = create_bidirectional_train_data(train_df, seed=seed)

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)

    if valid_df is None:
        if not 0.0 < validation_split_ratio < 1.0:
            raise ValueError("validation_split_ratio must be between 0 and 1 when valid_df is not provided.")
        split_dataset = train_dataset.train_test_split(test_size=validation_split_ratio, seed=seed)
        dataset_dict = DatasetDict({"train": split_dataset["train"], "validation": split_dataset["test"]})
    else:
        dataset_dict = DatasetDict(
            {
                "train": train_dataset,
                "validation": Dataset.from_pandas(valid_df, preserve_index=False),
            }
        )

    tokenize_fn: Callable[[Dict[str, list[str]]], Dict[str, Any]] = partial(
        tokenize_function,
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )

    tokenized_splits = {}
    for split_name, split_dataset in dataset_dict.items():
        tokenized_splits[split_name] = split_dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=split_dataset.column_names,
            desc=f"Tokenizing {split_name} dataset",
        )
    return DatasetDict(tokenized_splits)


def prepare_data_collator(
    tokenizer: PreTrainedTokenizerBase,
    model: Any,
) -> DataCollatorForSeq2Seq:
    """Create a seq2seq data collator with label padding masked as -100."""
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
    )
