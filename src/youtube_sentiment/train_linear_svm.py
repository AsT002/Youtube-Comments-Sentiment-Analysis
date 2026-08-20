"""Train Model C: word+character TF-IDF with a cross-validated Linear SVM."""

import argparse
from pathlib import Path
from time import perf_counter
from typing import Iterable

from .config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, ensure_project_directories
from .dataset import load_split
from .files import write_json
from .metrics import classification_metrics
from .text import SENTIMENT_TOKEN_PATTERN, normalize_text_lower


def build_features(word_features: int, character_features: int):
    """Build an unfitted feature union suitable for use inside a CV pipeline."""

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    preprocessor=normalize_text_lower,
                    tokenizer=None,
                    token_pattern=SENTIMENT_TOKEN_PATTERN,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=word_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    preprocessor=normalize_text_lower,
                    lowercase=False,
                    analyzer="char_wb",
                    ngram_range=(2, 6),
                    min_df=1,
                    max_features=character_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def build_svm_pipeline(c_value: float, word_features: int, character_features: int, seed: int):
    """Keep preprocessing inside the estimator so every CV fold fits it independently."""

    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    return Pipeline(
        [
            ("features", build_features(word_features, character_features)),
            ("classifier", LinearSVC(C=c_value, class_weight="balanced", random_state=seed)),
        ]
    )


def decision_scores(model, texts):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)
    import numpy as np

    scores = model.decision_function(texts)
    scores = np.asarray(scores, dtype=float)
    scores -= scores.max(axis=1, keepdims=True)
    exponentials = np.exp(scores)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-values", type=float, nargs="+", default=(0.25, 0.5, 1.0, 2.0))
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=1, help="Cross-validation workers; use 1 for bounded memory and portability.")
    parser.add_argument("--word-features", type=int, default=100_000)
    parser.add_argument("--character-features", type=int, default=100_000)
    parser.add_argument("--name", default="model_c_v2")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    try:
        import joblib
        import numpy as np
        from sklearn.metrics import f1_score, make_scorer
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except ImportError as exc:
        raise RuntimeError("Install the project training dependencies first.") from exc

    ensure_project_directories()
    output_dir = args.output_dir or ARTIFACTS_DIR / args.name
    train_texts, train_labels, _ = load_split(PROCESSED_DATA_DIR / "tweeteval" / "train.csv")
    validation_texts, validation_labels, _ = load_split(PROCESSED_DATA_DIR / "tweeteval" / "validation.csv")
    started = perf_counter()
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    scorer = make_scorer(f1_score, average="macro", zero_division=0)
    cv_results = []
    for c_value in args.c_values:
        estimator = build_svm_pipeline(c_value, args.word_features, args.character_features, args.seed)
        scores = cross_val_score(estimator, train_texts, train_labels, cv=cv, scoring=scorer, n_jobs=args.jobs)
        cv_results.append({"c": c_value, "fold_macro_f1": scores.tolist(), "mean_macro_f1": float(np.mean(scores)), "std_macro_f1": float(np.std(scores))})
        print(f"C={c_value:g}: CV macro F1={np.mean(scores):.4f} ± {np.std(scores):.4f}")
    best_result = max(cv_results, key=lambda result: result["mean_macro_f1"])
    model = build_svm_pipeline(best_result["c"], args.word_features, args.character_features, args.seed)
    model.fit(train_texts, train_labels)
    training_seconds = perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "model.joblib"
    joblib.dump(model, artifact)
    metrics = {"tweeteval": classification_metrics(validation_labels, decision_scores(model, validation_texts))}
    write_json(output_dir / "validation_metrics.json", metrics)
    write_json(
        output_dir / "metadata.json",
        {
            "name": args.name,
            "kind": "linear_svm",
            "artifact": str(artifact.resolve()),
            "random_seed": args.seed,
            "training_rows": len(train_texts),
            "training_seconds": training_seconds,
            "best_c": best_result["c"],
            "cross_validation": cv_results,
            "calibration": None,
            "feature_configuration": {
                "casefold": True,
                "word_ngram_range": [1, 2],
                "word_min_df": 2,
                "character_ngram_range": [2, 6],
                "character_min_df": 1,
                "preserve_emoji_and_punctuation": True,
            },
            "score_note": "Outputs are softmax-normalized SVM decision scores for ranking, not probabilities; confidence thresholds are disabled.",
        },
    )
    print(f"Saved {args.name} to {artifact}; TweetEval validation macro F1={metrics['tweeteval']['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
