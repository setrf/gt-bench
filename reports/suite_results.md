# GT-Bench Suite Results

Status: deterministic baselines and Tinker model evaluations are complete for the listed checkpoints.

Suite file: `data/suite/test.jsonl`
Suite SHA-256: `653aa5e1fd7bbcb2ff57e98c04cfc745a602f274c86dc8284bf875e650d1deee`

## Baselines

| Baseline | Accuracy | Correct | Incorrect |
| --- | ---: | ---: | ---: |
| `always_none` | 6.00% | 18 | 282 |
| `random` | 7.00% | 21 | 279 |
| `most_common_by_family` | 7.00% | 21 | 279 |
| `oracle` | 100.00% | 300 | 0 |

## Model Evaluations

| Checkpoint | Accuracy | Correct | Incorrect |
| --- | ---: | ---: | ---: |
| Base Qwen/Qwen3.6-27B | 20.00% | 60 | 240 |
| 2x2 SFT transfer | 48.33% | 145 | 155 |
| 2x2 + prompt-adversarial SFT transfer | 51.67% | 155 | 145 |
| Suite SFT | 68.00% | 204 | 96 |
| Joint base canonical+adv+suite | 84.67% | 254 | 46 |
| Joint adv-state suite+retention | 89.33% | 268 | 32 |
| Joint adv-state targeted+retention | 91.67% | 275 | 25 |
| Joint base full targeted | 93.00% | 279 | 21 |
| Base Qwen/Qwen3-8B | 4.00% | 12 | 288 |
| Base Qwen/Qwen3-30B-A3B | 6.33% | 19 | 281 |

## Canonical 2x2 Retention

| Checkpoint | Accuracy | Correct | Incorrect |
| --- | ---: | ---: | ---: |
| Suite SFT on canonical 2x2 | 53.60% | 268 | 232 |
| Joint base canonical+adv+suite on canonical 2x2 | 99.80% | 499 | 1 |
| Joint adv-state suite+retention on canonical 2x2 | 100.00% | 500 | 0 |
| Joint adv-state targeted+retention on canonical 2x2 | 99.80% | 499 | 1 |
| Joint base full targeted on canonical 2x2 | 97.20% | 486 | 14 |

## Accuracy By Task Family

### always_none

| Task family | Accuracy | Correct | Total |
| --- | ---: | ---: | ---: |
| `mixed_2x2` | 0.00% | 0 | 50 |
| `dominance` | 10.00% | 5 | 50 |
| `large_normal_form` | 4.00% | 2 | 50 |
| `extensive_form` | 12.00% | 6 | 50 |
| `natural_language` | 10.00% | 5 | 50 |
| `repeated_interaction` | 0.00% | 0 | 50 |

### random

| Task family | Accuracy | Correct | Total |
| --- | ---: | ---: | ---: |
| `mixed_2x2` | 0.00% | 0 | 50 |
| `dominance` | 2.00% | 1 | 50 |
| `large_normal_form` | 4.00% | 2 | 50 |
| `extensive_form` | 16.00% | 8 | 50 |
| `natural_language` | 20.00% | 10 | 50 |
| `repeated_interaction` | 0.00% | 0 | 50 |

### most_common_by_family

| Task family | Accuracy | Correct | Total |
| --- | ---: | ---: | ---: |
| `mixed_2x2` | 0.00% | 0 | 50 |
| `dominance` | 10.00% | 5 | 50 |
| `large_normal_form` | 4.00% | 2 | 50 |
| `extensive_form` | 12.00% | 6 | 50 |
| `natural_language` | 14.00% | 7 | 50 |
| `repeated_interaction` | 2.00% | 1 | 50 |

### oracle

| Task family | Accuracy | Correct | Total |
| --- | ---: | ---: | ---: |
| `mixed_2x2` | 100.00% | 50 | 50 |
| `dominance` | 100.00% | 50 | 50 |
| `large_normal_form` | 100.00% | 50 | 50 |
| `extensive_form` | 100.00% | 50 | 50 |
| `natural_language` | 100.00% | 50 | 50 |
| `repeated_interaction` | 100.00% | 50 | 50 |

## Model Accuracy By Task Family

| Checkpoint | `mixed_2x2` | `dominance` | `large_normal_form` | `extensive_form` | `natural_language` | `repeated_interaction` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base Qwen/Qwen3.6-27B | 0.00% | 0.00% | 16.00% | 36.00% | 26.00% | 42.00% |
| 2x2 SFT transfer | 0.00% | 32.00% | 72.00% | 66.00% | 54.00% | 66.00% |
| 2x2 + prompt-adversarial SFT transfer | 0.00% | 32.00% | 70.00% | 80.00% | 60.00% | 68.00% |
| Suite SFT | 78.00% | 44.00% | 54.00% | 76.00% | 56.00% | 100.00% |
| Joint base canonical+adv+suite | 80.00% | 56.00% | 82.00% | 92.00% | 98.00% | 100.00% |
| Joint adv-state suite+retention | 92.00% | 60.00% | 86.00% | 98.00% | 100.00% | 100.00% |
| Joint adv-state targeted+retention | 96.00% | 68.00% | 86.00% | 100.00% | 100.00% | 100.00% |
| Joint base full targeted | 98.00% | 78.00% | 84.00% | 98.00% | 100.00% | 100.00% |
| Base Qwen/Qwen3-8B | 0.00% | 0.00% | 14.00% | 0.00% | 10.00% | 0.00% |
| Base Qwen/Qwen3-30B-A3B | 0.00% | 0.00% | 8.00% | 0.00% | 28.00% | 2.00% |
