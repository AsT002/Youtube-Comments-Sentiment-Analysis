"""Download the declared TweetEval source dataset without running training."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

from .config import RAW_DATA_DIR, TWEETEVAL_CONFIG, TWEETEVAL_DATASET, ensure_project_directories
from .files import sha256_file, write_json, write_jsonl


def download_tweeteval(output_dir: Path) -> Dict[str, object]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package before downloading TweetEval.") from exc

    dataset = load_dataset(TWEETEVAL_DATASET, TWEETEVAL_CONFIG)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    checksums: Dict[str, str] = {}
    for split in ("train", "validation", "test"):
        destination = output_dir / f"{split}.jsonl"
        counts[split] = write_jsonl(destination, dataset[split])
        checksums[destination.name] = sha256_file(destination)

    metadata: Dict[str, object] = {
        "dataset": TWEETEVAL_DATASET,
        "config": TWEETEVAL_CONFIG,
        "dataset_page": "https://huggingface.co/datasets/cardiffnlp/tweet_eval",
        "license": "CC BY 3.0 (sentiment subset; upstream platform terms may also apply)",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "fingerprints": {split: dataset[split]._fingerprint for split in ("train", "validation", "test")},
        "sha256": checksums,
    }
    write_json(output_dir / "source.json", metadata)
    return metadata


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("tweeteval",))
    parser.parse_args(argv)
    ensure_project_directories()
    metadata = download_tweeteval(RAW_DATA_DIR / "tweeteval")
    print(f"Downloaded TweetEval: {metadata['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
