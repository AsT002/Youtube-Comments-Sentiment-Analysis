"""Train the TF-IDF logistic-regression baseline."""

import argparse
from pathlib import Path
from time import perf_counter
from typing import Iterable

from .config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, ensure_project_directories
from .dataset import load_split
from .files import write_json
from .metrics import classification_metrics
from .text import normalize_text_lower


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--name", default="baseline_v2")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError("Install the project training dependencies first.") from exc

    ensure_project_directories()
    output_dir = args.output_dir or ARTIFACTS_DIR / args.name
    train_texts, train_labels, _ = load_split(PROCESSED_DATA_DIR / "tweeteval" / "train.csv")
    validation_texts, validation_labels, _ = load_split(PROCESSED_DATA_DIR / "tweeteval" / "validation.csv")
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(preprocessor=normalize_text_lower, lowercase=False, tokenizer=None, ngram_range=(1, 2), min_df=2, max_features=args.max_features, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=RANDOM_SEED)),
        ]
    )
    started = perf_counter()
    model.fit(train_texts, train_labels)
    training_seconds = perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "model.joblib"
    joblib.dump(model, artifact)

    metrics = {"tweeteval": classification_metrics(validation_labels, model.predict_proba(validation_texts))}
    write_json(output_dir / "validation_metrics.json", metrics)
    write_json(output_dir / "metadata.json", {"name": args.name, "kind": "baseline", "artifact": str(artifact.resolve()), "training_seconds": training_seconds, "random_seed": RANDOM_SEED, "training_rows": len(train_texts), "max_features": args.max_features, "casefold": True})
    print(f"Saved baseline to {artifact}; TweetEval validation macro F1={metrics['tweeteval']['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
