"""Shared project configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
CACHE_DIR = PROJECT_ROOT / "cache"

RANDOM_SEED = 42
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_IDS = {name: label for label, name in LABEL_NAMES.items()}
CANDIDATE_NAMES = ("baseline_v2", "model_c_v2")

TWEETEVAL_DATASET = "cardiffnlp/tweet_eval"
TWEETEVAL_CONFIG = "sentiment"


def ensure_project_directories() -> None:
    """Create directories used for generated data and artifacts."""

    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        ARTIFACTS_DIR,
        REPORTS_DIR,
        CACHE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
