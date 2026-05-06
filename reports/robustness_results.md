# GT-Bench Robustness Results

Robustness set: `data/robust/robust_seed271828.jsonl`
Robustness SHA-256: `ffc6c26df85c78df702d4eef910b835025a5b2a8344a9ef1d14e1ab7e78405e8`

The robustness set keeps the same 2x2 pure-strategy Nash task but changes the prompt surface. It contains 250 examples: 5 prompt variants x 5 equilibrium-count buckets x 10 examples.

## Overall Accuracy

| Run | Accuracy | 95% exact binomial CI | Correct | Incorrect | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 64.00% | 57.71%-69.95% | 160 | 90 | 0.00 pp |
| 5000-example SFT | 88.40% | 83.77%-92.09% | 221 | 29 | +24.40 pp |

## Accuracy By Prompt Variant

| Prompt variant | Baseline | 5000-example SFT | Delta |
| --- | ---: | ---: | ---: |
| `answer_only` | 28.00% | 100.00% | +72.00 pp |
| `compact_pairs` | 64.00% | 78.00% | +14.00 pp |
| `json_payoffs` | 56.00% | 68.00% | +12.00 pp |
| `minimal_matrix` | 90.00% | 96.00% | +6.00 pp |
| `standard_table` | 82.00% | 100.00% | +18.00 pp |

## Accuracy By Number Of Equilibria

| Equilibria | Baseline | 5000-example SFT | Delta |
| ---: | ---: | ---: | ---: |
| 0 | 38.00% | 90.00% | +52.00 pp |
| 1 | 78.00% | 100.00% | +22.00 pp |
| 2 | 72.00% | 88.00% | +16.00 pp |
| 3 | 44.00% | 80.00% | +36.00 pp |
| 4 | 88.00% | 84.00% | -4.00 pp |

## Fine-Tuned Failure Preview

- `robust_compact_pairs_eq0_0005` (`compact_pairs`)
  - Gold: `[]`
  - Prediction: Final answer: The pure-strategy Nash equilibrium is (U, L).
- `robust_compact_pairs_eq0_0006` (`compact_pairs`)
  - Gold: `[]`
  - Prediction: Final answer: The pure-strategy Nash equilibrium is (D, L).
- `robust_compact_pairs_eq2_0002` (`compact_pairs`)
  - Gold: `[['D', 'R'], ['U', 'L']]`
  - Prediction: Final answer: The pure-strategy Nash equilibrium is (U, L).
- `robust_compact_pairs_eq2_0007` (`compact_pairs`)
  - Gold: `[['D', 'R'], ['U', 'L']]`
  - Prediction: Final answer: The pure-strategy Nash equilibrium is (D, R).
- `robust_compact_pairs_eq3_0003` (`compact_pairs`)
  - Gold: `[['D', 'R'], ['U', 'L'], ['U', 'R']]`
  - Prediction: Final answer: The pure-strategy Nash equilibria are (U, L) and (D, R).
- `robust_compact_pairs_eq3_0004` (`compact_pairs`)
  - Gold: `[['D', 'R'], ['U', 'L'], ['U', 'R']]`
  - Prediction: Final answer: The pure-strategy Nash equilibria are (U, L), (U, R), (D, L), and (D, R).
- `robust_compact_pairs_eq3_0005` (`compact_pairs`)
  - Gold: `[['D', 'L'], ['U', 'L'], ['U', 'R']]`
  - Prediction: Final answer: The pure-strategy Nash equilibria are (U, L) and (D, L).
- `robust_compact_pairs_eq3_0008` (`compact_pairs`)
  - Gold: `[['D', 'R'], ['U', 'L'], ['U', 'R']]`
  - Prediction: Final answer: The pure-strategy Nash equilibria are (U, L) and (D, R).
- `robust_compact_pairs_eq4_0005` (`compact_pairs`)
  - Gold: `[['D', 'L'], ['D', 'R'], ['U', 'L'], ['U', 'R']]`
  - Prediction: Final answer: The pure-strategy Nash equilibria are (U, L), (D, L), and (D, R).
- `robust_compact_pairs_eq4_0006` (`compact_pairs`)
  - Gold: `[['D', 'L'], ['D', 'R'], ['U', 'L'], ['U', 'R']]`
  - Prediction: Final answer: The pure-strategy Nash equilibria are (U, L), (D, L), and (D, R).

## Interpretation

The fine-tuned checkpoint improves overall robustness, which makes simple prompt-template overfitting less likely. The weakest remaining variants should drive the next targeted adversarial SFT slice.
