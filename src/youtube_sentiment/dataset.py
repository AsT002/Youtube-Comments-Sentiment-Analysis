"""Load processed sentiment splits without framework-specific dependencies."""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from .text import normalize_label, normalize_text


def load_split(path: Path) -> Tuple[List[str], List[int], List[Dict[str, str]]]:
    texts: List[str] = []
    labels: List[int] = []
    metadata: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = normalize_text(row.get("text"))
            label = normalize_label(row.get("label"))
            if not text or label is None:
                continue
            texts.append(text)
            labels.append(label)
            metadata.append({"source": row.get("source", ""), "video_id": row.get("video_id", ""), "split": row.get("split", "")})
    return texts, labels, metadata
