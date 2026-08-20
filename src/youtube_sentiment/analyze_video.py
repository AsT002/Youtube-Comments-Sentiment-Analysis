"""Analyze live top-level YouTube comments with the selected local model."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Iterable

from .artifacts import load_artifact, predict_class_scores
from .config import ARTIFACTS_DIR, CACHE_DIR, LABEL_NAMES
from .youtube_api import YouTubeAPIError, fetch_comments, load_cached_comments, parse_video_id, save_cached_comments


def load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def summarize(scores, score_threshold=None):
    import numpy as np

    values = np.asarray(scores)
    predicted = values.argmax(axis=1)
    counts = {name: int((predicted == label).sum()) for label, name in LABEL_NAMES.items()}
    result = {
        "comments": len(values),
        "counts": counts,
        "percentages": {name: count * 100 / len(values) for name, count in counts.items()},
        "mean_class_scores": {LABEL_NAMES[label]: float(values[:, label].mean()) for label in range(3)},
    }
    if score_threshold is not None:
        result["low_score_comments"] = int((values.max(axis=1) < score_threshold).sum())
        result["score_threshold"] = score_threshold
    return result


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="YouTube video ID or URL")
    parser.add_argument("--manifest", type=Path, default=ARTIFACTS_DIR / "production.json")
    parser.add_argument("--api-key", help="Overrides YOUTUBE_API_KEY from the environment")
    parser.add_argument("--max-comments", type=int, default=1_000, help="Use 0 to retrieve all top-level comments")
    parser.add_argument("--cache-hours", type=float, default=24.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--score-threshold", type=float, help="Optional threshold for calibrated models; disabled for uncalibrated SVM scores.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)
    if args.score_threshold is not None and not 0 < args.score_threshold <= 1:
        parser.error("--score-threshold must be in (0, 1]")
    if args.max_comments < 0:
        parser.error("--max-comments cannot be negative")

    try:
        load_environment()
        video_id = parse_video_id(args.video)
        cache_scope = "all" if args.max_comments == 0 else str(args.max_comments)
        cache_path = CACHE_DIR / f"{video_id}_{cache_scope}.json"
        comments = None if args.refresh else load_cached_comments(cache_path, video_id, timedelta(hours=args.cache_hours))
        if comments is None:
            api_key = args.api_key or os.getenv("YOUTUBE_API_KEY") or os.getenv("API_KEY")
            comments = fetch_comments(api_key or "", video_id, None if args.max_comments == 0 else args.max_comments)
            save_cached_comments(cache_path, video_id, comments)
        if not comments:
            print("No accessible top-level comments were found for this video.")
            return 0

        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        kind, model = load_artifact(Path(manifest["artifact"]))
        uncalibrated_svm = kind == "sklearn" and not hasattr(model, "predict_proba")
        if uncalibrated_svm and args.score_threshold is not None:
            parser.error("--score-threshold cannot be used with an uncalibrated SVM model")
        score_note = manifest.get("score_note", manifest.get("confidence_note"))
        result = {"video_id": video_id, "model": manifest.get("name"), "score_note": score_note, **summarize(predict_class_scores(kind, model, comments), args.score_threshold)}
    except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, ValueError, YouTubeAPIError) as exc:
        parser.exit(1, f"error: {exc}\n")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Analyzed {result['comments']} top-level comments with {result['model']}.")
        for label in ("positive", "neutral", "negative"):
            print(f"{label.title()}: {result['counts'][label]} ({result['percentages'][label]:.2f}%)")
        if "low_score_comments" in result:
            print(f"Low score: {result['low_score_comments']} (< {result['score_threshold']:.0%})")
        means = result["mean_class_scores"]
        print(f"Mean class scores: positive={means['positive']:.3f}, neutral={means['neutral']:.3f}, negative={means['negative']:.3f}")
        if result.get("score_note"):
            print(f"Score note: {result['score_note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
