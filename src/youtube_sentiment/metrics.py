"""Consistent three-class evaluation metrics."""

from typing import Dict, Sequence

from .config import LABEL_NAMES


def classification_metrics(y_true: Sequence[int], class_scores) -> Dict[str, object]:
    try:
        import numpy as np
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    except ImportError as exc:
        raise RuntimeError("Install numpy and scikit-learn before evaluating models.") from exc

    score_array = np.asarray(class_scores)
    if score_array.ndim != 2 or score_array.shape[1] != 3:
        raise ValueError(f"Expected an N x 3 class-score matrix, received {score_array.shape}")
    predictions = score_array.argmax(axis=1)
    labels = [0, 1, 2]
    report = classification_report(
        y_true,
        predictions,
        labels=labels,
        target_names=[LABEL_NAMES[label] for label in labels],
        output_dict=True,
        zero_division=0,
    )
    return {
        "examples": len(y_true),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "per_class": {name: report[name] for name in LABEL_NAMES.values()},
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
    }
