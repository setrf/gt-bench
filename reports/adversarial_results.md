# GT-Bench Adversarial SFT Follow-Up

Status: `complete`

This follow-up keeps the mathematical task unchanged and targets the prompt-format weakness exposed by the robustness set. The adversarial training supplement adds 500 conservative prompt-variant examples, emphasizing compact payoff pairs and JSON-like payoff objects while preserving balanced equilibrium-count buckets.

## Accuracy Summary

| Evaluation | Original 5000 SFT | Adversarial SFT | Delta |
| --- | ---: | ---: | ---: |
| canonical | 99.60% | 99.80% | +0.20 pp |
| confirmation | 99.70% | 99.90% | +0.20 pp |
| stress | 100.00% | 100.00% | +0.00 pp |
| robustness | 88.40% | 98.80% | +10.40 pp |

## Robustness By Prompt Variant

| Prompt variant | Original 5000 SFT | Adversarial SFT | Delta |
| --- | ---: | ---: | ---: |
| `answer_only` | 100.00% | 100.00% | +0.00 pp |
| `compact_pairs` | 78.00% | 100.00% | +22.00 pp |
| `json_payoffs` | 68.00% | 94.00% | +26.00 pp |
| `minimal_matrix` | 96.00% | 100.00% | +4.00 pp |
| `standard_table` | 100.00% | 100.00% | +0.00 pp |

## Acceptance Criteria

- Target robustness accuracy: at least 95.00%.
- Canonical regression tolerance: no worse than -0.50 percentage points versus the original 5000-example SFT checkpoint.
- Main expected gains: `compact_pairs` and `json_payoffs`.

## Claim Boundaries

This follow-up can support a targeted robustness claim for one synthetic formal task. It does not claim broad game-theory reasoning, and it still excludes mixed strategies, dominance, welfare, sequential games, and natural-language story problems.
