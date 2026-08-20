"""Perform the one-time final evaluation of the selected production model."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

from .artifacts import load_artifact, predict_class_scores
from .config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from .dataset import load_split
from .files import write_json
from .metrics import classification_metrics


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("tweeteval",))
    parser.add_argument("--manifest", type=Path, default=ARTIFACTS_DIR / "production.json")
    parser.add_argument("--confirm-final-test", action="store_true", help="Required acknowledgement that this opens a sealed test split.")
    args = parser.parse_args(argv)
    if not args.confirm_final_test:
        parser.error("Final test evaluation requires --confirm-final-test")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / f"final_{args.dataset}_{manifest['name']}_metrics.json"
    existing_reports = sorted(REPORTS_DIR.glob(f"final_{args.dataset}*_metrics.json"))
    if existing_reports:
        parser.error(f"The {args.dataset} test split was already opened; refusing another evaluation. Existing report: {existing_reports[0]}")
    split_path = PROCESSED_DATA_DIR / args.dataset / "test.csv"
    texts, labels, _ = load_split(split_path)
    kind, model = load_artifact(Path(manifest["artifact"]))
    started = perf_counter()
    scores = predict_class_scores(kind, model, texts)
    elapsed = perf_counter() - started
    metrics = classification_metrics(labels, scores)
    metrics["prediction_seconds"] = elapsed
    metrics["examples_per_second"] = len(texts) / elapsed if elapsed else None
    metrics["interpretation"] = "trusted benchmark evaluation"
    write_json(output, metrics)
    test_reports = dict(manifest.get("test_reports", {}))
    test_reports[args.dataset] = str(output.resolve())
    manifest["test_reports"] = test_reports
    manifest["test_results_used"] = True
    write_json(args.manifest, manifest)
    print(f"{args.dataset} test macro F1={metrics['macro_f1']:.4f}; wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
