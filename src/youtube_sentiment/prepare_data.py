"""Normalize TweetEval while preserving its official splits."""

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_project_directories
from .files import read_jsonl, write_json
from .text import normalize_label, normalize_text

FIELDS = ("text", "label", "source", "video_id", "split")


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
            count += 1
    return count


def prepare_tweeteval(raw_dir: Path, output_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split in ("train", "validation", "test"):
        rows: List[Dict[str, object]] = []
        for source_row in read_jsonl(raw_dir / f"{split}.jsonl"):
            text = normalize_text(source_row.get("text"))
            label = normalize_label(source_row.get("label"))
            if text and label is not None:
                rows.append({"text": text, "label": label, "source": "tweeteval", "video_id": "", "split": split})
        counts[split] = write_csv(output_dir / f"{split}.csv", rows)
    write_json(output_dir / "metadata.json", {"source": "tweeteval", "counts": counts, "labels": {"0": "negative", "1": "neutral", "2": "positive"}})
    return counts


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("tweeteval",))
    parser.parse_args(argv)
    ensure_project_directories()
    counts = prepare_tweeteval(RAW_DATA_DIR / "tweeteval", PROCESSED_DATA_DIR / "tweeteval")
    print(f"Prepared TweetEval: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
