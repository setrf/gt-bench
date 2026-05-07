# GT-Bench Adversarial SFT Follow-Up

Status: `pending`

This follow-up keeps the mathematical task unchanged and targets the prompt-format weakness exposed by the robustness set. The adversarial training supplement emphasizes compact payoff pairs, JSON-like payoff objects, answer-only prompts, minimal matrices, and balanced equilibrium-count buckets.

The public pipeline is implemented, but the new Tinker SFT/evaluation run has not been completed in this checkout yet.

## Accuracy Summary

| Evaluation | Original 5000 SFT | Adversarial SFT | Delta |
| --- | ---: | ---: | ---: |
| canonical | 99.60% | pending | pending |
| confirmation | 99.70% | pending | pending |
| stress | 100.00% | pending | pending |
| robustness | 88.40% | pending | pending |

## Robustness By Prompt Variant

| Prompt variant | Original 5000 SFT | Adversarial SFT | Delta |
| --- | ---: | ---: | ---: |
| `answer_only` | 100.00% | pending | pending |
| `compact_pairs` | 78.00% | pending | pending |
| `json_payoffs` | 68.00% | pending | pending |
| `minimal_matrix` | 96.00% | pending | pending |
| `standard_table` | 100.00% | pending | pending |

## Acceptance Criteria

- Target robustness accuracy: at least 95.00%.
- Canonical regression tolerance: no worse than -0.50 percentage points versus the original 5000-example SFT checkpoint.
- Main expected gains: `compact_pairs` and `json_payoffs`.

## Claim Boundaries

This follow-up can support a targeted robustness claim for one synthetic formal task. It does not claim broad game-theory reasoning, and it still excludes mixed strategies, dominance, welfare, sequential games, and natural-language story problems.
