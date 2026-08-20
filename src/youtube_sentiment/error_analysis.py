"""Summarize validation errors without exposing or tuning on test data."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .artifacts import load_artifact, predict_class_scores
from .config import ARTIFACTS_DIR, LABEL_NAMES, PROCESSED_DATA_DIR, REPORTS_DIR
from .dataset import load_split

NEGATION_RE = re.compile(r"\b(?:not|no|never|neither|nor|hardly|isn't|wasn't|don't|didn't|can't|won't)\b", re.IGNORECASE)
ELONGATED_RE = re.compile(r"([A-Za-z])\1{2,}", re.IGNORECASE)


def feature_flags(text: str):
    return {
        "contains_negation": bool(NEGATION_RE.search(text)),
        "contains_emoji_or_non_ascii": any(not character.isascii() for character in text),
        "contains_url": "<URL>" in text,
        "contains_user_mention": "<USER>" in text,
        "very_short_3_words_or_less": len(text.split()) <= 3,
        "repeated_punctuation": "!!" in text or "??" in text,
        "elongated_word": bool(ELONGATED_RE.search(text)),
    }


def render_report(model_name: str, texts, labels, predictions) -> str:
    total = len(labels)
    correct = sum(actual == predicted for actual, predicted in zip(labels, predictions))
    confusion = Counter((actual, predicted) for actual, predicted in zip(labels, predictions) if actual != predicted)
    feature_totals = Counter()
    feature_errors = Counter()
    for text, actual, predicted in zip(texts, labels, predictions):
        for feature, present in feature_flags(text).items():
            if present:
                feature_totals[feature] += 1
                feature_errors[feature] += actual != predicted

    confusion_rows = "\n".join(
        f"| {LABEL_NAMES[actual]} | {LABEL_NAMES[predicted]} | {count:,} |"
        for (actual, predicted), count in confusion.most_common()
    )
    feature_rows = "\n".join(
        f"| {feature.replace('_', ' ')} | {count:,} | {feature_errors[feature] / count:.2%} |"
        for feature, count in feature_totals.most_common()
    )
    return f"""# Validation error analysis

This report uses TweetEval validation data only. It contains aggregate statistics rather than source text.

## Model

- Candidate: `{model_name}`
- Validation examples: {total:,}
- Correct: {correct:,}
- Error rate: {(total - correct) / total:.2%}

## Misclassification directions

| Actual | Predicted | Count |
| --- | --- | ---: |
{confusion_rows}

## Error rate for text characteristics

These categories overlap and do not establish causation.

| Characteristic | Examples | Error rate |
| --- | ---: | ---: |
{feature_rows}

## Interpretation

Use these aggregates to propose future training features or architectures. Do not change the current candidate based on test-set behavior; the test split remains outside this analysis.
"""


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ARTIFACTS_DIR / "production.json")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "validation_error_analysis.md")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    texts, labels, _ = load_split(PROCESSED_DATA_DIR / "tweeteval" / "validation.csv")
    kind, model = load_artifact(Path(manifest["artifact"]))
    scores = predict_class_scores(kind, model, texts)
    predictions = scores.argmax(axis=1).tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(str(manifest["name"]), texts, labels, predictions), encoding="utf-8")
    print(f"Wrote validation error analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
