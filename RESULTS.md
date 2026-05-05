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

## Interpretation

The strongest improvement is on zero-equilibrium cases. On the canonical split, the baseline solved only 9 of 45 zero-equilibrium games after parser correction, while the 1000-example and 5000-example fine-tunes solved all 45.

The remaining failures in the best run are tie-heavy cases where the model over-predicts an extra equilibrium.

## What This Does And Does Not Show

This shows that targeted Tinker SFT can measurably improve a model on a narrow, fully verifiable formal reasoning task.

It does not show broad game-theory competence. The benchmark covers only 2x2 normal-form games, pure equilibria, integer payoffs, and synthetic prompts.
