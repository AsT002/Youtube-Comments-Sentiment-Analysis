"""Load scikit-learn model artifacts through a common score interface."""

import json
from pathlib import Path
from typing import Sequence

from .text import normalize_text


def load_artifact(path: Path):
    if path.suffix != ".joblib":
        raise ValueError(f"Unsupported artifact type: {path}")
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Install joblib and scikit-learn to load the sentiment model.") from exc
    return "sklearn", joblib.load(path)


def predict_class_scores(kind: str, model, texts: Sequence[str], batch_size: int = 256):
    del batch_size  # Retained for call-site compatibility.
    if kind != "sklearn":
        raise ValueError(f"Unsupported model kind: {kind}")
    normalized = [normalize_text(text) for text in texts]
    import numpy as np

    classes = list(model.classes_)
    if classes != [0, 1, 2]:
        raise ValueError(f"Scikit-learn artifact has unexpected classes: {classes}")
    if hasattr(model, "predict_proba"):
        return model.predict_proba(normalized)
    scores = np.asarray(model.decision_function(normalized), dtype=float)
    scores -= scores.max(axis=1, keepdims=True)
    exponentials = np.exp(scores)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def production_artifact(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return Path(manifest["artifact"])
