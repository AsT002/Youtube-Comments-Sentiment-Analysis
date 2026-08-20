# Validation error analysis

This report uses TweetEval validation data only. It contains aggregate statistics rather than source text.

## Model

- Candidate: `model_c_v2`
- Validation examples: 2,000
- Correct: 1,412
- Error rate: 29.40%

## Misclassification directions

| Actual | Predicted | Count |
| --- | --- | ---: |
| positive | neutral | 166 |
| neutral | positive | 153 |
| neutral | negative | 105 |
| negative | neutral | 95 |
| positive | negative | 43 |
| negative | positive | 26 |

## Error rate for text characteristics

These categories overlap and do not establish causation.

| Characteristic | Examples | Error rate |
| --- | ---: | ---: |
| contains user mention | 620 | 31.45% |
| contains negation | 332 | 37.05% |
| repeated punctuation | 98 | 22.45% |
| elongated word | 26 | 46.15% |
| contains url | 5 | 20.00% |

## Interpretation

Use these aggregates to propose future training features or architectures. Do not change the current candidate based on test-set behavior; the test split remains outside this analysis.
