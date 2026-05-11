# GT-Bench Suite Results

Status: local deterministic suite smoke baselines are complete; Tinker model evaluations are pending.

Suite file: `data/suite/test.jsonl`
Suite SHA-256: `8f7ea7c9df92d02efdf2258ec8b1c5cd83e005b6c75bb351b1397803a301cd03`

## Baselines

| Baseline | Accuracy | Correct | Incorrect |
| --- | ---: | ---: | ---: |
| `always_none` | 6.00% | 18 | 282 |
| `random` | 5.67% | 17 | 283 |
| `most_common_by_family` | 6.67% | 20 | 280 |
| `oracle` | 100.00% | 300 | 0 |

## Accuracy By Task Family

### always_none

| Task family | Accuracy | Correct | Total |
| --- | ---: | ---: | ---: |
| `mixed_2x2` | 0.00% | 0 | 50 |
| `dominance` | 4.00% | 2 | 50 |
| `large_normal_form` | 14.00% | 7 | 50 |
| `extensive_form` | 10.00% | 5 | 50 |
| `natural_language` | 8.00% | 4 | 50 |
| `repeated_interaction` | 0.00% | 0 | 50 |

### random

| Task family | Accuracy | Correct | Total |
| --- | ---: | ---: | ---: |
| `mixed_2x2` | 0.00% | 0 | 50 |
| `dominance` | 6.00% | 3 | 50 |
| `large_normal_form` | 6.00% | 3 | 50 |
| `extensive_form` | 12.00% | 6 | 50 |
| `natural_language` | 10.00% | 5 | 50 |
| `repeated_interaction` | 0.00% | 0 | 50 |

### most_common_by_family

| Task family | Accuracy | Correct | Total |
| --- | ---: | ---: | ---: |
| `mixed_2x2` | 4.00% | 2 | 50 |
| `dominance` | 2.00% | 1 | 50 |
| `large_normal_form` | 12.00% | 6 | 50 |
| `extensive_form` | 10.00% | 5 | 50 |
| `natural_language` | 12.00% | 6 | 50 |
| `repeated_interaction` | 0.00% | 0 | 50 |

### oracle

| Task family | Accuracy | Correct | Total |
| --- | ---: | ---: | ---: |
| `mixed_2x2` | 100.00% | 50 | 50 |
| `dominance` | 100.00% | 50 | 50 |
| `large_normal_form` | 100.00% | 50 | 50 |
| `extensive_form` | 100.00% | 50 | 50 |
| `natural_language` | 100.00% | 50 | 50 |
| `repeated_interaction` | 100.00% | 50 | 50 |

## Pending Model Evaluations

- base `Qwen/Qwen3.6-27B` on the suite test split
- current 2x2 pure-equilibrium SFT checkpoint on the suite test split
- full-suite SFT checkpoint
- optional adversarial full-suite checkpoint

No model-result claim should be made for the broader suite until these reports exist.
