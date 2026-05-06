# GT-Bench Qwen3.6-27B Results

Model: `Qwen/Qwen3.6-27B`
Test set: `data/test.jsonl`
Test SHA-256: `a9af74346a7b67cc13952fc858570dee6d03e4248b04e830c42f8c1bb935d219`

## Exact-Match Accuracy

| Run | Accuracy | 95% exact binomial CI | Correct | Incorrect | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 87.60% | 84.39%-90.36% | 438 | 62 | 0.00 pp |
| qwen36_27b_sft_0250 | 53.40% | 48.92%-57.84% | 267 | 233 | -34.20 pp |
| qwen36_27b_sft_1000 | 91.60% | 88.82%-93.88% | 458 | 42 | +4.00 pp |
| qwen36_27b_sft_5000 | 99.60% | 98.56%-99.95% | 498 | 2 | +12.00 pp |

## Accuracy By Number Of Equilibria

### baseline

| Equilibria | Total | Correct | Accuracy |
| ---: | ---: | ---: | ---: |
| 0 | 45 | 9 | 20.00% |
| 1 | 308 | 287 | 93.18% |
| 2 | 141 | 137 | 97.16% |
| 3 | 6 | 5 | 83.33% |

### qwen36_27b_sft_0250

| Equilibria | Total | Correct | Accuracy |
| ---: | ---: | ---: | ---: |
| 0 | 45 | 0 | 0.00% |
| 1 | 308 | 230 | 74.68% |
| 2 | 141 | 37 | 26.24% |
| 3 | 6 | 0 | 0.00% |

### qwen36_27b_sft_1000

| Equilibria | Total | Correct | Accuracy |
| ---: | ---: | ---: | ---: |
| 0 | 45 | 45 | 100.00% |
| 1 | 308 | 281 | 91.23% |
| 2 | 141 | 127 | 90.07% |
| 3 | 6 | 5 | 83.33% |

### qwen36_27b_sft_5000

| Equilibria | Total | Correct | Accuracy |
| ---: | ---: | ---: | ---: |
| 0 | 45 | 45 | 100.00% |
| 1 | 308 | 308 | 100.00% |
| 2 | 141 | 139 | 98.58% |
| 3 | 6 | 6 | 100.00% |

## Independent Confirmation

Confirmation set: `data/confirm/confirm_seed20260505.jsonl`
Confirmation SHA-256: `d9874764ce086f23065568ba5066745e0a879955a924e8a257d2f3b1f82a0cd8`

| Run | Accuracy | 95% exact binomial CI | Correct | Incorrect | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 89.20% | 87.11%-91.06% | 892 | 108 | 0.00 pp |
| qwen36_27b_sft_5000 | 99.70% | 99.13%-99.94% | 997 | 3 | +10.50 pp |

## Balanced Stress Evaluation

Stress set: `data/stress/tie_stress_seed314159.jsonl`
Stress SHA-256: `9c298c756940bb57189f114da491f8b56b57dd9bedddefe82e61c82669f94340`

The stress set contains equal numbers of examples with 0, 1, 2, 3, and 4 pure-strategy equilibria.

| Run | Accuracy | 95% exact binomial CI | Correct | Incorrect | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 81.60% | 76.23%-86.20% | 204 | 46 | 0.00 pp |
| qwen36_27b_sft_5000 | 100.00% | 98.54%-100.00% | 250 | 0 | +18.40 pp |

## Stress Accuracy By Number Of Equilibria

### baseline

| Equilibria | Total | Correct | Accuracy |
| ---: | ---: | ---: | ---: |
| 0 | 50 | 16 | 32.00% |
| 1 | 50 | 49 | 98.00% |
| 2 | 50 | 50 | 100.00% |
| 3 | 50 | 40 | 80.00% |
| 4 | 50 | 49 | 98.00% |

### qwen36_27b_sft_5000

| Equilibria | Total | Correct | Accuracy |
| ---: | ---: | ---: | ---: |
| 0 | 50 | 50 | 100.00% |
| 1 | 50 | 50 | 100.00% |
| 2 | 50 | 50 | 100.00% |
| 3 | 50 | 50 | 100.00% |
| 4 | 50 | 50 | 100.00% |

## Takeaway

Best fine-tuned run: `qwen36_27b_sft_5000` at 99.60%.

## Best Run Failure Preview

- `example_005625`
  - Gold: `[['D', 'R'], ['U', 'R']]`
  - Prediction: Final answer: The pure-strategy Nash equilibria are (U, R), (D, L), and (D, R).
- `example_005990`
  - Gold: `[['U', 'L'], ['U', 'R']]`
  - Prediction: Final answer: The pure-strategy Nash equilibria are (U, L), (U, R), and (D, L).

## Limitations

This is a narrow synthetic benchmark for 2x2 pure-strategy Nash equilibria only. It does not demonstrate broad game-theory reasoning improvement.
