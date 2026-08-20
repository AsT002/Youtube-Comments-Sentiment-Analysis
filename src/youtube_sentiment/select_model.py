"""Select the strongest TweetEval-only production candidate."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from .config import ARTIFACTS_DIR, CANDIDATE_NAMES, PROJECT_ROOT
from .files import write_json


def load_candidate(directory: Path) -> Optional[Dict[str, object]]:
    metadata_path = directory / "metadata.json"
    metrics_path = directory / "validation_metrics.json"
    if not metadata_path.exists() or not metrics_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    score_note = metadata.get("score_note", metadata.get("confidence_note"))
    return {"name": metadata["name"], "kind": metadata["kind"], "artifact": metadata["artifact"], "metrics": metrics, "score_note": score_note}


def choose_candidate(candidates) -> Dict[str, object]:
    if not candidates:
        raise RuntimeError("Train at least one TweetEval candidate before selection.")
    best = max(candidates, key=lambda item: item["metrics"]["tweeteval"]["macro_f1"])
    return {**best, "selection_reason": "Highest TweetEval validation macro F1."}


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=PROJECT_ROOT / "config" / "experiment.json")
    parser.add_argument("--output", type=Path, default=ARTIFACTS_DIR / "production.json")
    args = parser.parse_args(argv)
    experiment = json.loads(args.experiment.read_text(encoding="utf-8"))
    candidates = [candidate for name in CANDIDATE_NAMES for candidate in [load_candidate(ARTIFACTS_DIR / name)] if candidate is not None]
    selected = choose_candidate(candidates)
    manifest = {"name": selected["name"], "kind": selected["kind"], "artifact": selected["artifact"], "selection_reason": selected["selection_reason"], "validation_metrics": selected["metrics"], "selection_rule": experiment, "score_note": selected.get("score_note"), "test_results_used": False}
    write_json(args.output, manifest)
    print(f"Selected {selected['name']}: {selected['selection_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
