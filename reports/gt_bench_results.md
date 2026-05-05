# GT-Bench Qwen3.6-27B Results

Model: `Qwen/Qwen3.6-27B`
Test set: `data/test.jsonl`
Test SHA-256: `a9af74346a7b67cc13952fc858570dee6d03e4248b04e830c42f8c1bb935d219`

## Exact-Match Accuracy

| Run | Accuracy | Correct | Incorrect | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 87.60% | 438 | 62 | 0.00 pp |
| qwen36_27b_sft_0250 | 53.40% | 267 | 233 | -34.20 pp |
| qwen36_27b_sft_1000 | 91.60% | 458 | 42 | +4.00 pp |
| qwen36_27b_sft_5000 | 99.60% | 498 | 2 | +12.00 pp |

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

| Run | Accuracy | Correct | Incorrect | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 89.20% | 892 | 108 | 0.00 pp |
| qwen36_27b_sft_5000 | 99.70% | 997 | 3 | +10.50 pp |

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
