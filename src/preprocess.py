"""Text preprocessing utilities for Akkadian MT."""

from __future__ import annotations

import re
from typing import Pattern

_WHITESPACE_RE: Pattern[str] = re.compile(r"\s+")
_GAP_RE: Pattern[str] = re.compile(
    r"(?:\.{3,}|\[\.{1,}\]|<\.\.\.|x{2,}|(?:\bx\b[\s\-]*){2,}|\b(?:x\s*){2,}\b)",
    flags=re.IGNORECASE,
)
_DECIMAL_RE: Pattern[str] = re.compile(r"(?<!\d)(\d+)\.(\d+)(?!\d)")


def ascii_to_diacritics(text: str) -> str:
    """Convert common ASCII transliteration sequences to Akkadian diacritics."""
    replacements = (
        (r"(?i)sz", "š"),
        (r"s,", "ṣ"),
        (r"t,", "ṭ"),
        (r"h,", "ḫ"),
        (r"S,", "Ṣ"),
        (r"T,", "Ṭ"),
        (r"H,", "Ḫ"),
    )
    normalized = text
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def normalize_gaps(text: str) -> str:
    """Map common missing-text markers to a shared <gap> token."""
    normalized = _GAP_RE.sub(" <gap> ", text)
    normalized = re.sub(r"(?:<gap>\s*){2,}", "<gap> ", normalized)
    return normalized


def canon_decimal(text: str) -> str:
    """Normalize decimal numbers by stripping redundant trailing zeros."""

    def _replace(match: re.Match[str]) -> str:
        integer, decimal = match.groups()
        decimal = decimal.rstrip("0")
        return integer if not decimal else f"{integer}.{decimal}"

    return _DECIMAL_RE.sub(_replace, text)


def preprocess_akkadian_text(text: str) -> str:
    """Apply lightweight normalization for Akkadian transliteration."""
    normalized = str(text or "")
    normalized = ascii_to_diacritics(normalized)
    normalized = normalize_gaps(normalized)
    normalized = canon_decimal(normalized)
    normalized = normalized.replace("\u00a0", " ")
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def preprocess_english_text(text: str) -> str:
    """Apply lightweight normalization for English targets."""
    normalized = str(text or "")
    normalized = normalized.replace("\u2019", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = normalized.replace("\u00a0", " ")
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized
