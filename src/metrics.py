"""Evaluation metrics for Akkadian MT."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import evaluate
import numpy as np
from transformers import EvalPrediction, PreTrainedTokenizerBase

from .preprocess import preprocess_english_text

_BLEU = evaluate.load("sacrebleu")
_CHRF = evaluate.load("chrf")


def postprocess_text(predictions: Iterable[str], references: Iterable[str]) -> Tuple[List[str], List[List[str]]]:
    """Normalize whitespace and package references for evaluate."""
    preds = [preprocess_english_text(pred) for pred in predictions]
    refs = [[preprocess_english_text(ref)] for ref in references]
    return preds, refs


def build_compute_metrics(tokenizer: PreTrainedTokenizerBase):
    """Create a compute_metrics callback for Seq2SeqTrainer."""

    def compute_metrics(eval_prediction: EvalPrediction) -> Dict[str, float]:
        predictions = eval_prediction.predictions
        labels = eval_prediction.label_ids

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
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
