"""Evaluation metrics for Akkadian MT."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import evaluate
import numpy as np
from transformers import EvalPrediction, PreTrainedTokenizerBase

from .preprocess import postprocess_english_batch, preprocess_english_batch

_BLEU = evaluate.load("sacrebleu")
_CHRF = evaluate.load("chrf")


def prepare_prediction_ids(predictions, tokenizer: PreTrainedTokenizerBase) -> np.ndarray:
    """Normalize trainer predictions into valid token ids for decoding."""
    prediction_array = predictions[0] if isinstance(predictions, tuple) else predictions
    prediction_array = np.asarray(prediction_array)

    # Some transformers versions return logits even with generate-based evaluation.
    if prediction_array.ndim == 3:
        prediction_array = prediction_array.argmax(axis=-1)

    prediction_array = prediction_array.astype(np.int64, copy=False)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    vocab_upper_bound = len(tokenizer) - 1
    invalid_mask = (prediction_array < 0) | (prediction_array > vocab_upper_bound)
    if invalid_mask.any():
        prediction_array = prediction_array.copy()
        prediction_array[invalid_mask] = pad_token_id
    return prediction_array


def decode_prediction_batch(predictions, tokenizer: PreTrainedTokenizerBase) -> List[str]:
    """Decode trainer predictions safely across transformers versions."""
    prediction_ids = prepare_prediction_ids(predictions, tokenizer)
    return tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)


def postprocess_text(predictions: Iterable[str], references: Iterable[str]) -> Tuple[List[str], List[List[str]]]:
    """Normalize whitespace and package references for evaluate."""
    preds = postprocess_english_batch(list(predictions))
    refs = [[text] for text in preprocess_english_batch(list(references))]
    return preds, refs


def build_compute_metrics(tokenizer: PreTrainedTokenizerBase):
    """Create a compute_metrics callback for Seq2SeqTrainer."""

    def compute_metrics(eval_prediction: EvalPrediction) -> Dict[str, float]:
        labels = eval_prediction.label_ids

        decoded_predictions = decode_prediction_batch(eval_prediction.predictions, tokenizer)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_predictions, decoded_labels = postprocess_text(decoded_predictions, decoded_labels)

        bleu_result = _BLEU.compute(predictions=decoded_predictions, references=decoded_labels)
        chrf_result = _CHRF.compute(predictions=decoded_predictions, references=decoded_labels, word_order=2)

        bleu = float(bleu_result["score"])
        chrf = float(chrf_result["score"])
        combined_score = math.sqrt(max(bleu, 0.0) * max(chrf, 0.0))

        return {
            "bleu": round(bleu, 4),
            "chrf++": round(chrf, 4),
            "combined_score": round(combined_score, 4),
        }

    return compute_metrics
