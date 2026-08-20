"""Evaluate every TweetEval-only candidate on the same validation split."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

from .artifacts import load_artifact, predict_class_scores
from .config import ARTIFACTS_DIR, CANDIDATE_NAMES, PROCESSED_DATA_DIR
from .dataset import load_split
from .files import write_json
from .metrics import classification_metrics


def validation_metrics(artifact: Path, dataset: Path):
    kind, model = load_artifact(artifact)
    texts, labels, _ = load_split(dataset)
    started = perf_counter()
    scores = predict_class_scores(kind, model, texts)
    elapsed = perf_counter() - started
    metrics = classification_metrics(labels, scores)
    metrics["prediction_seconds"] = elapsed
    metrics["examples_per_second"] = len(texts) / elapsed if elapsed else None
    return {"tweeteval": metrics}


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    validation = PROCESSED_DATA_DIR / "tweeteval" / "validation.csv"
    evaluated = 0
    for name in CANDIDATE_NAMES:
        directory = ARTIFACTS_DIR / name
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        artifact = Path(metadata["artifact"])
        results = validation_metrics(artifact, validation)
        results["artifact_size_bytes"] = artifact.stat().st_size if artifact.is_file() else None
        write_json(directory / "validation_metrics.json", results)
        evaluated += 1
        print(f"{name}: TweetEval validation macro F1={results['tweeteval']['macro_f1']:.4f}")
    if not evaluated:
        raise RuntimeError("No trained candidate artifacts were found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
