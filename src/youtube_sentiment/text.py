"""Text and label normalization shared by data and inference code."""

import re
from typing import Optional

from .config import LABEL_IDS, LABEL_NAMES

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
USER_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
WHITESPACE_RE = re.compile(r"\s+")
SENTIMENT_TOKEN_PATTERN = r"(?u)<url>|<user>|\b\w+\b|[^\w\s]"
SENTIMENT_TOKEN_RE = re.compile(SENTIMENT_TOKEN_PATTERN)


def normalize_text(value: object) -> str:
    """Normalize transport-specific tokens while preserving sentiment signals."""

    if value is None:
        return ""
    text = str(value).strip()
    text = URL_RE.sub("<URL>", text)
    text = USER_RE.sub("<USER>", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_text_lower(value: object) -> str:
    """Normalize text and case-fold it without discarding punctuation or emoji."""

    return normalize_text(value).casefold()


def sentiment_tokenize(value: str):
    """Tokenize words, emoji, and sentiment-bearing punctuation explicitly."""

    return SENTIMENT_TOKEN_RE.findall(value)


def normalize_label(value: object) -> Optional[int]:
    """Convert known numeric or textual sentiment labels to 0, 1, or 2."""

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value in LABEL_NAMES:
        return value

    text = str(value).strip().lower()
    aliases = {
        "0": 0,
        "negative": 0,
        "neg": 0,
        "1": 1,
        "neutral": 1,
        "neu": 1,
        "2": 2,
        "positive": 2,
        "pos": 2,
    }
    return aliases.get(text, LABEL_IDS.get(text))


def is_probably_english(text: str) -> bool:
    """Apply a cheap pre-filter before optional language identification."""

    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(character.isascii() for character in letters)
    return ascii_letters / len(letters) >= 0.8
