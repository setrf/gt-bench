# GT-Bench Results

GT-Bench is a minimal Tinker fine-tuning benchmark for one formal reasoning task: given a random 2x2 two-player normal-form payoff matrix, find all pure-strategy Nash equilibria.

The demo uses `Qwen/Qwen3.6-27B` on Tinker with LoRA SFT. The benchmark is deliberately narrow, synthetic, and exactly scored.

## Main Result

On the canonical 500-example held-out test split, the 5000-example fine-tuned checkpoint improved exact-match accuracy from 87.60% to 99.60%.

| Run | Accuracy | Correct | Incorrect | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 87.60% | 438 | 62 | 0.00 pp |
| 250-example SFT | 53.40% | 267 | 233 | -34.20 pp |
| 1000-example SFT | 91.60% | 458 | 42 | +4.00 pp |
| 5000-example SFT | 99.60% | 498 | 2 | +12.00 pp |

The 250-example run made the model worse, which is useful evidence that too little targeted data can distort the learned rule. The 1000-example run improved the model modestly, and the 5000-example run nearly solved the task.

## Confirmation Result

To check that the result was not a one-split fluke, I generated a fresh 1000-example confirmation set with seed `20260505` and evaluated the original base model and the best 5000-example checkpoint.

| Run | Accuracy | Correct | Incorrect | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 89.20% | 892 | 108 | 0.00 pp |
| 5000-example SFT | 99.70% | 997 | 3 | +10.50 pp |

The confirmation result supports the main claim: targeted fine-tuning produced a large, measurable improvement on held-out 2x2 pure-strategy Nash equilibrium problems.

## Balanced Stress Result

To probe the hardest cases more directly, I generated a 250-example stress set with exactly 50 games each containing 0, 1, 2, 3, and 4 pure-strategy equilibria.

| Run | Accuracy | Correct | Incorrect | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 81.60% | 204 | 46 | 0.00 pp |
| 5000-example SFT | 100.00% | 250 | 0 | +18.40 pp |

The baseline remained strong on ordinary one- and two-equilibrium games but solved only 16 of 50 zero-equilibrium games. The 5000-example fine-tuned checkpoint solved every bucket: 0, 1, 2, 3, and 4 equilibria.

## Prompt Robustness Result

To check whether the model was simply matching the original prompt template, I evaluated on a 250-example robustness set with five prompt variants.

| Run | Accuracy | Correct | Incorrect | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 64.00% | 160 | 90 | 0.00 pp |
| 5000-example SFT | 88.40% | 221 | 29 | +24.40 pp |

The fine-tuned checkpoint improved substantially, but it was not fully prompt-invariant. The weakest remaining variants were compact payoff pairs and JSON-like payoff objects.

## Adversarial Follow-Up

The adversarial SFT pipeline now targets that prompt-format gap directly. It adds a deterministic 1000-example supplement to the 5000-example training set, weighted toward compact payoff pairs and JSON-like payoff objects while balancing every prompt variant across equilibrium-count buckets.

The follow-up result tracker is `ADVERSARIAL_SFT.md` and `reports/adversarial_results.md`. Until the bounded Tinker run is complete, the public tracker is marked pending rather than substituting a different model or checkpoint.

## Interpretation

The strongest improvement is on zero-equilibrium cases. On the canonical split, the baseline solved only 9 of 45 zero-equilibrium games after parser correction, while the 1000-example and 5000-example fine-tunes solved all 45.

The canonical and confirmation failures in the best run are tie-heavy cases where the model over-predicts an extra equilibrium. The balanced stress result suggests that this residual weakness is rare, but still worth tracking because it is exactly the kind of edge case a narrow formal benchmark can expose. The robustness result adds one more useful finding: prompt format matters, so targeted data should include compact and structured payoff presentations.

## What This Does And Does Not Show

This shows that targeted Tinker SFT can measurably improve a model on a narrow, fully verifiable formal reasoning task.

It does not show broad game-theory competence. The benchmark covers only 2x2 normal-form games, pure equilibria, integer payoffs, and synthetic prompts.
