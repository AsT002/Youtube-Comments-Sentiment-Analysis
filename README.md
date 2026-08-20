# YouTube Comments Sentiment Analysis

Train and compare three-class sentiment models that classify text as **negative**, **neutral**, or **positive**. The project uses TweetEval only and does not collect comments for training.

This implementation replaces the original IMDb prototype because movie reviews used the wrong domain, binary labels, and incompatible preprocessing.

## Dataset

[TweetEval sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval) is the sole training and benchmark source:

- Training: 45,615 examples
- Validation: 2,000 examples
- Test: 12,284 examples
- Labels: negative, neutral, positive

The official splits are preserved. Live YouTube comments are analyzed only at inference time and are never added to training data.

## Setup

Requirements:

- Python 3.9–3.12
- A YouTube Data API v3 key only for live analysis

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Reproducible workflow

### 1. Download and prepare TweetEval

```bash
sentiment-download tweeteval
sentiment-prepare tweeteval
```

Generated splits are written under `data/processed/tweeteval/`.

### 2. Train candidates

```bash
sentiment-train-baseline
sentiment-train-svm
```

The candidates are:

- Baseline v2: word TF-IDF plus logistic regression
- Model C v2: case-folded word/character TF-IDF plus Linear SVM

Model C v2 preserves emoji and punctuation features and fits the entire TF-IDF/SVM pipeline independently inside each cross-validation fold.

### What the models were trained on

Both maintained models were trained only on TweetEval's 45,615-example training split. The 2,000-example validation split was used to compare the finished candidates; it was not added to training. The 12,284-example test split was not used to fit or select Model C v2. YouTube comments are used only when running the analyzer and are never added to either model.

### How long training took

On the machine used for this experiment:

- Baseline v2 trained in **3.1 seconds**.
- Model C v2 took **114.6 seconds**, or about **1 minute 55 seconds**, including hyperparameter selection and the final fit.

These are measured wall-clock times, so they will vary by CPU, memory, operating system, and scikit-learn version. Model C v2 is a Linear SVM and therefore does not train for a chosen number of epochs. It optimizes until its convergence tolerance is reached or its 1,000-iteration safety limit is hit. The final saved SVM converged after 30 optimization iterations.

The longer Model C v2 time is intentional: four regularization values (`C = 0.25, 0.5, 1.0, 2.0`) were each evaluated across three folds, producing 12 isolated training runs, and the winning `C = 0.25` configuration was then fitted once on the complete training split. This was long enough to compare regularization honestly without fitting TF-IDF features across fold boundaries.

### 3. Validate and select

```bash
sentiment-validate
sentiment-select
sentiment-error-analysis
```

Selection uses TweetEval validation macro F1 only and writes `artifacts/production.json`.

### Current result

[The experiment report](reports/experiment_summary.md) selects Model C v2:

- Honest three-fold CV macro F1: **0.6557 ± 0.0012**
- Validation macro F1: **0.6825**
- Validation accuracy: **70.60%**

Model C v2 has no final test score. TweetEval test was already opened for historical v1 releases, so the project now blocks another evaluation.

### 4. Final-evaluation policy

In a fresh reproduction where no final report exists, evaluate the frozen winner once:

```bash
sentiment-evaluate tweeteval --confirm-final-test
```

The guard is global: once any final TweetEval report exists, later candidates cannot reopen the split.

## Analyze a YouTube video

Copy the environment template and add your key:

```bash
cp .env.example .env
```

Analyze a URL or video ID:

```bash
sentiment-analyze 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
sentiment-analyze VIDEO_ID --max-comments 500
sentiment-analyze VIDEO_ID --refresh
sentiment-analyze VIDEO_ID --json
```

The analyzer validates IDs, paginates with retries and timeouts, caches fetched comments for 24 hours, and handles empty or disabled comment sections. Model C v2 reports normalized SVM class scores—not probabilities—and disables confidence thresholds.

## Tests

```bash
python -m pytest
ruff check .
```

Tests cover normalization, sentiment tokens, fold-safe SVM construction, label mapping, production selection, global test sealing, YouTube URL parsing, pagination, and cache behavior.

## Known limitations

- TweetEval contains tweets rather than YouTube comments.
- Without a human-labeled YouTube test set, real YouTube accuracy is unknown.
- Model C v2 scores are uncalibrated SVM decision scores, not probabilities.
- Sarcasm, code-switching, and references to video context remain difficult.
- The current pipeline is English-only.
