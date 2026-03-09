"""Text preprocessing and postprocessing utilities for Akkadian MT."""

from __future__ import annotations

import math
import re
from typing import List, Sequence

_V2_RE = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3_RE = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE_TRANS = str.maketrans({"a": "á", "e": "é", "i": "í", "u": "ú", "A": "Á", "E": "É", "I": "Í", "U": "Ú"})
_GRAVE_TRANS = str.maketrans({"a": "à", "e": "è", "i": "ì", "u": "ù", "A": "À", "E": "È", "I": "Ì", "U": "Ù"})
_ALLOWED_FRACS = [
    (1 / 6, "0.16666"),
    (1 / 4, "0.25"),
    (1 / 3, "0.33333"),
    (1 / 2, "0.5"),
    (2 / 3, "0.66666"),
    (3 / 4, "0.75"),
    (5 / 6, "0.83333"),
]
_FRAC_TOL = 2e-3
_FLOAT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")
_WHITESPACE_RE = re.compile(r"\s+")
_GAP_UNIFIED_RE = re.compile(
    r"<\s*big[\s_\-]*gap\s*>"
    r"|<\s*gap\s*>"
    r"|\bbig[\s_\-]*gap\b"
    r"|\bx(?:\s+x)+\b"
    r"|\.{3,}|…+|\[\.+\]"
    r"|\[\s*x\s*\]|\(\s*x\s*\)"
    r"|(?<!\w)x{2,}(?!\w)"
    r"|(?<!\w)x(?!\w)"
    r"|\(\s*large\s+break\s*\)"
    r"|\(\s*break\s*\)"
    r"|\(\s*\d+\s+broken\s+lines?\s*\)",
    re.IGNORECASE,
)
_UNICODE_UPPER = r"A-ZŠṬṢḪ\u00C0-\u00D6\u00D8-\u00DE\u0160\u1E00-\u1EFF"
_UNICODE_LOWER = r"a-zšṭṣḫ\u00E0-\u00F6\u00F8-\u00FF\u0161\u1E01-\u1EFF"
_DET_UPPER_RE = re.compile(r"\(([" + _UNICODE_UPPER + r"0-9]{1,6})\)")
_DET_LOWER_RE = re.compile(r"\(([" + _UNICODE_LOWER + r"]{1,4})\)")
_PN_RE = re.compile(r"\bPN\b")
_KUBABBAR_RE = re.compile(r"KÙ\.B\.")
_MONTH_RE = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.IGNORECASE)
_ROMAN2INT = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12}
_SOFT_GRAM_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)",
    re.IGNORECASE,
)
_QUOTES_RE = re.compile(r"[\u201c\u201d\u2018\u2019\"']")
_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_REPEAT_PUNCT_RE = re.compile(r"([.,])\1+")
_PUNCT_SPACE_RE = re.compile(r"\s+([.,:])")
_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {
    "0.8333": "⅚",
    "0.6666": "⅔",
    "0.3333": "⅓",
    "0.1666": "⅙",
    "0.625": "⅝",
    "0.75": "¾",
    "0.25": "¼",
    "0.5": "½",
}
_SUB_X = "ₓ"
_CHAR_TRANS = str.maketrans(
    {
        "ḫ": "h",
        "Ḫ": "H",
        "ʾ": "",
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "—": "-",
        "–": "-",
        "\u00a0": " ",
    }
)
_FORBIDDEN_TRANS = str.maketrans("", "", '()——<>⌈⌋⌊[]+ʾ;!')


def ascii_to_diacritics(text: str) -> str:
    """Convert common ASCII transliteration sequences into diacritic forms."""
    normalized = str(text or "")
    normalized = normalized.replace("sz", "š").replace("SZ", "Š")
    normalized = normalized.replace("s,", "ṣ").replace("S,", "Ṣ")
    normalized = normalized.replace("t,", "ṭ").replace("T,", "Ṭ")
    normalized = normalized.replace("h,", "ḫ").replace("H,", "Ḫ")
    normalized = _V2_RE.sub(lambda match: match.group(1).translate(_ACUTE_TRANS), normalized)
    normalized = _V3_RE.sub(lambda match: match.group(1).translate(_GRAVE_TRANS), normalized)
    return normalized


def _canon_decimal_number(value: float) -> str:
    integer_part = int(math.floor(value + 1e-12))
    fractional = value - integer_part
    best = min(_ALLOWED_FRACS, key=lambda item: abs(fractional - item[0]))
    if abs(fractional - best[0]) <= _FRAC_TOL:
        decimal = best[1]
        if integer_part == 0:
            return decimal
        return f"{integer_part}{decimal[1:]}" if decimal.startswith("0.") else f"{integer_part}+{decimal}"
    return f"{value:.5f}".rstrip("0").rstrip(".")


def canon_decimal(text: str) -> str:
    """Normalize decimal strings into a canonical compact form."""
    normalized = str(text or "")
    return _FLOAT_RE.sub(lambda match: _canon_decimal_number(float(match.group(1))), normalized)


def normalize_gaps(text: str) -> str:
    """Normalize gap markers and x-runs into a shared <gap> token."""
    normalized = str(text or "")
    normalized = _GAP_UNIFIED_RE.sub("<gap>", normalized)
    normalized = re.sub(r"(?:\s*<gap>\s*){2,}", " <gap> ", normalized)
    return normalized


def _replace_exact_fraction(match: re.Match[str]) -> str:
    return _EXACT_FRAC_MAP[match.group(0)]


def preprocess_akkadian_text(text: str) -> str:
    """Apply the stronger transliteration normalization used for inference."""
    normalized = ascii_to_diacritics(text)
    normalized = _DET_UPPER_RE.sub(r"\1", normalized)
    normalized = _DET_LOWER_RE.sub(r"{\1}", normalized)
    normalized = normalize_gaps(normalized)
    normalized = normalized.translate(_CHAR_TRANS)
    normalized = normalized.replace(_SUB_X, "")
    normalized = _KUBABBAR_RE.sub("KÙ.BABBAR", normalized)
    normalized = _EXACT_FRAC_RE.sub(_replace_exact_fraction, normalized)
    normalized = canon_decimal(normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def _collapse_gap_runs(text: str) -> str:
    tokens = text.split()
    collapsed: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index] == "<gap>":
            while index < len(tokens) and tokens[index] == "<gap>":
                index += 1
            collapsed.append("<gap>")
        else:
            collapsed.append(tokens[index])
            index += 1
    return " ".join(collapsed)


def _month_replace(match: re.Match[str]) -> str:
    roman = match.group(1).upper()
    return f"Month {_ROMAN2INT.get(roman, roman)}"


def preprocess_english_text(text: str) -> str:
    """Light normalization for target English text before tokenization."""
    normalized = str(text or "")
    normalized = normalized.replace("\u2019", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalize_gaps(normalized)
    normalized = canon_decimal(normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def postprocess_english_text(text: str) -> str:
    """Postprocess generated English text with the same rules used in notebook inference."""
    normalized = str(text or "")
    normalized = normalize_gaps(normalized)
    normalized = _PN_RE.sub("<gap>", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    normalized = _SOFT_GRAM_RE.sub(" ", normalized)
    normalized = _QUOTES_RE.sub("", normalized)
    normalized = _collapse_gap_runs(normalized)
    normalized = normalized.replace("<gap>", "\x00GAP\x00")
    normalized = normalized.translate(_FORBIDDEN_TRANS)
    normalized = normalized.replace("\x00GAP\x00", " <gap> ")
    normalized = canon_decimal(normalized)
    normalized = _MONTH_RE.sub(_month_replace, normalized)
    normalized = _REPEAT_WORD_RE.sub(r"\1", normalized)
    for ngram_size in range(4, 1, -1):
        pattern = re.compile(r"\b((?:\w+\s+){" + str(ngram_size - 1) + r"}\w+)(?:\s+\1\b)+")
        normalized = pattern.sub(r"\1", normalized)
    normalized = _PUNCT_SPACE_RE.sub(r"\1", normalized)
    normalized = _REPEAT_PUNCT_RE.sub(r"\1", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def preprocess_akkadian_batch(texts: Sequence[str]) -> List[str]:
    """Apply Akkadian preprocessing to a batch of texts."""
    return [preprocess_akkadian_text(text) for text in texts]



def preprocess_english_batch(texts: Sequence[str]) -> List[str]:
    """Apply target-side preprocessing to a batch of texts."""
    return [preprocess_english_text(text) for text in texts]



def postprocess_english_batch(texts: Sequence[str]) -> List[str]:
    """Apply generated-text postprocessing to a batch of texts."""
    return [postprocess_english_text(text) for text in texts]
