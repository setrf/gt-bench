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

The adversarial SFT pipeline targets that prompt-format gap directly. It adds a deterministic 500-example supplement to the 5000-example training set, weighted toward compact payoff pairs and JSON-like payoff objects while balancing every prompt variant across equilibrium-count buckets.

The follow-up result met the acceptance criteria: canonical accuracy increased from 99.60% to 99.80%, and prompt-robustness accuracy increased from 88.40% to 98.80%. The weakest original variants improved from 78.00% to 100.00% for compact payoff pairs and from 68.00% to 94.00% for JSON-like payoff objects.

The public follow-up summary is `reports/adversarial_results.md`.

## Repeated-Seed Learning Curve

I repeated the SFT training-size sweep across three independent training-data seeds while keeping the canonical 500-example test set fixed. Seed `42` is the original sweep; seeds `1009` and `2027` were newly generated.

| Train size | Mean accuracy | Seed SD | Mean delta vs baseline |
| ---: | ---: | ---: | ---: |
| 250 | 50.60% | 3.86 pp | -37.00 pp |
| 1000 | 84.67% | 12.53 pp | -2.93 pp |
| 5000 | 99.60% | 0.40 pp | +12.00 pp |

The key result is that 5000-example SFT is stable across seeds: 99.60%, 99.20%, and 100.00%. The 1000-example condition is not stable; two seeds improved over baseline, but seed `2027` fell to 70.20%. The 250-example condition consistently underperformed baseline.

The public repeated-seed summary is `reports/seed_sweep_results.md`.

## Interpretation

The strongest improvement is on zero-equilibrium cases. On the canonical split, the baseline solved only 9 of 45 zero-equilibrium games under the exact prediction parser, while the 1000-example and 5000-example fine-tunes solved all 45.

The canonical and confirmation failures in the best run are tie-heavy cases where the model over-predicts an extra equilibrium. The balanced stress result suggests that this residual weakness is rare, but still worth tracking because it is exactly the kind of edge case a narrow formal benchmark can expose. The robustness result adds one more useful finding: prompt format matters, so targeted data should include compact and structured payoff presentations. The repeated-seed result adds a second caveat: 1000 examples can be enough, but that regime is seed-sensitive; 5000 examples is the stable setting in this experiment.

## What This Does And Does Not Show

This shows that targeted Tinker SFT can measurably improve a model on a narrow, fully verifiable formal reasoning task.

It does not show broad game-theory competence. These results cover only 2x2 normal-form games, pure equilibria, integer payoffs, and synthetic prompts. The repository now includes a broader exactly scored suite for mixed strategies, dominance, larger normal-form games, extensive form, natural-language descriptions, and repeated interaction, but no model-result claim is attached to that suite yet.
