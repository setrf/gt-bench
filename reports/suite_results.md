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

## Canonical 2x2 Retention

| Checkpoint | Accuracy | Correct | Incorrect |
| --- | ---: | ---: | ---: |
| Suite SFT on canonical 2x2 | 53.60% | 268 | 232 |

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
