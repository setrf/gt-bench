# GT-Bench Repeated-Seed Learning Curve

Status: `complete`
Model: `Qwen/Qwen3.6-27B`
Fixed test set: `data/test.jsonl`

This summary repeats the SFT training-size sweep across independent training-data seeds while keeping the canonical 500-example test set fixed. Seed `42` is the original public sweep; the additional seeds test whether the learning curve is stable under regenerated synthetic training data.

## Per-Seed Accuracy

| Train size | Seed | Accuracy | Correct | Incorrect | Delta vs baseline |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 42 | 53.40% | 267 | 233 | -34.20 pp |
| 250 | 1009 | 46.20% | 231 | 269 | -41.40 pp |
| 250 | 2027 | 52.20% | 261 | 239 | -35.40 pp |
| 1000 | 42 | 91.60% | 458 | 42 | +4.00 pp |
| 1000 | 1009 | 92.20% | 461 | 39 | +4.60 pp |
| 1000 | 2027 | 70.20% | 351 | 149 | -17.40 pp |
| 5000 | 42 | 99.60% | 498 | 2 | +12.00 pp |
| 5000 | 1009 | 99.20% | 496 | 4 | +11.60 pp |
| 5000 | 2027 | 100.00% | 500 | 0 | +12.40 pp |

## Aggregate Statistics

| Train size | n | Mean accuracy | Seed SD | 95% t CI for mean accuracy | Mean delta vs baseline | 95% bootstrap CI for mean delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 3 | 50.60% | 3.86 pp | 41.02-60.18% | -37.00 pp | -41.40 to -34.20 pp |
| 1000 | 3 | 84.67% | 12.53 pp | 53.53-100.00% | -2.93 pp | -17.40 to +4.60 pp |
| 5000 | 3 | 99.60% | 0.40 pp | 98.61-100.00% | +12.00 pp | +11.60 to +12.40 pp |

## Paired Training-Size Comparisons

| Comparison | n | Mean delta | 95% t CI | 95% bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| 1000 vs 250 | 3 | +34.07 pp | -1.83 to +69.97 pp | +18.00 to +46.00 pp |
| 5000 vs 1000 | 3 | +14.93 pp | -17.08 to +46.94 pp | +7.00 to +29.80 pp |
| 5000 vs 250 | 3 | +49.00 pp | +40.17 to +57.83 pp | +46.20 to +53.00 pp |

## Interpretation

This is a seed-level comparison, not a new task. It supports a stronger learning-curve claim when the 1000- and 5000-example means remain above baseline across independently generated training sets. With only three seeds, uncertainty intervals should be read as descriptive rather than definitive.
