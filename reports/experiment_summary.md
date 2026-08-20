# Experiment report

## Current conclusion

The corrected production candidate is **Model C v2**, a Linear SVM with case-folded word and character TF-IDF features. It achieved **0.6825 macro F1** and **70.60% accuracy** on the 2,000-example TweetEval validation split.

Model C v2 has not been evaluated on TweetEval test. That split was already opened for historical releases and is now globally sealed against further evaluation.

## Data and metric

TweetEval is the sole dataset:

- Training: 45,615 examples
- Validation: 2,000 examples
- Test: 12,284 examples

Macro F1 is the primary metric because it gives negative, neutral, and positive equal importance despite class imbalance. Accuracy is reported as a secondary metric.

Both maintained candidates were fitted only on the 45,615-example training split. Validation examples were used to compare completed candidates, not to update their parameters. TweetEval test and live YouTube comments were not used to train or select Model C v2.

## Training duration

On the experiment machine, Baseline v2 trained in **3.1 seconds**. Model C v2 took **114.6 seconds** (about **1 minute 55 seconds**) for hyperparameter selection and its final fit. Wall-clock duration is hardware- and environment-dependent and should be treated as a reproducibility reference, not a fixed stopping rule.

Linear SVMs do not use neural-network epochs. Each fit optimized until convergence or the configured 1,000-iteration safety limit. The final Model C v2 fit converged after 30 optimization iterations. Its total time includes 12 cross-validation fits—four `C` values across three independent folds—followed by one final fit on all training examples. This procedure was chosen to select regularization while keeping TF-IDF vocabulary and IDF estimation isolated inside each fold.

## Candidate results

| Candidate | Status | Validation macro F1 | Validation accuracy |
| --- | --- | ---: | ---: |
| Baseline v1 | Historical; unintentionally case-sensitive | 0.6371 | 65.80% |
| Model A | Historical; unintentionally case-sensitive BiLSTM | 0.5917 | 60.25% |
| Model C v1 | Historical; CV preprocessing leak | 0.6687 | 69.60% |
| Model D | Historical; unintentionally case-sensitive CNN | 0.6175 | 62.95% |
| Baseline v2 | Corrected case-folding | 0.6424 | 66.25% |
| **Model C v2** | **Current production candidate** | **0.6825** | **70.60%** |

Model C v2 improves validation macro F1 by 0.0401 over Baseline v2 and 0.0138 over Model C v1. A fixed-seed, 5,000-resample paired bootstrap against Baseline v2 produced a 95% interval of approximately **[0.0196, 0.0602]** for the improvement.

Only Baseline v2 and Model C v2 remain in the maintained training pipeline. The older artifacts and TensorFlow training scripts were removed after the comparison; their measurements remain here as historical experiment results.

## Methodology correction

The first experiment round supplied custom sklearn and TensorFlow preprocessors without explicit lowercasing. Model C v1 also fitted TF-IDF vocabulary and IDF weights before cross-validation.

Model C v2 fixes those issues by:

- Case-folding explicitly
- Preserving emoji and punctuation as word and character features
- Fitting the complete TF-IDF/SVM pipeline independently inside every fold
- Saving a new artifact rather than overwriting the historically tested model

## Honest cross-validation

| SVM `C` | Mean CV macro F1 | Fold standard deviation |
| ---: | ---: | ---: |
| **0.25** | **0.6557** | 0.0012 |
| 0.50 | 0.6475 | 0.0002 |
| 1.00 | 0.6358 | 0.0014 |
| 2.00 | 0.6282 | 0.0009 |

The more strongly regularized `C=0.25` setting generalized best. The validation result is 0.0267 above mean cross-validation, so 0.6825 should not be treated as expected deployment accuracy.

## Historical test results

These results describe old artifacts only:

| Historical release | Accuracy | Macro F1 | Negative F1 | Neutral F1 | Positive F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline v1 | 0.5860 | 0.5818 | 0.5864 | 0.5965 | 0.5625 |
| Model C v1 | 0.6103 | 0.6058 | 0.5921 | 0.6286 | 0.5967 |

TweetEval changes substantially between development and test: negative prevalence rises from 15.55% in training and 15.60% in validation to 32.33% in test, while mean whitespace-token length falls from 19.24/19.44 to 14.86. This establishes distribution shift but does not prove which characteristic caused the historical decline.

## Current validation errors

Model C v2 made 588 errors on 2,000 validation examples. Its largest error directions were positive to neutral (166), neutral to positive (153), neutral to negative (105), and negative to neutral (95). Negation-bearing text had a 37.05% error rate versus 29.40% overall. See [validation_error_analysis.md](validation_error_analysis.md).

## Recommendation and limits

Use Model C v2 as the current research prototype. It has the strongest honest cross-validation and TweetEval validation results. Its softmax-normalized SVM margins are ranking scores, not calibrated probabilities.

Do not claim a final test result for Model C v2 or verified YouTube accuracy. Real deployment accuracy remains unknown without an independently human-labeled YouTube evaluation set.
