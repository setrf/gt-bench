# GT-Bench Technical Report

## Summary

GT-Bench is a compact Tinker fine-tuning experiment for one formal reasoning task: given a 2x2 two-player normal-form payoff matrix, identify all pure-strategy Nash equilibria.

The project was built by Mert Gulsun, a UC Berkeley master's student and Thinking Machines Lab Tinker research grant recipient. The goal is to show whether targeted fine-tuning can measurably improve a model on a narrow, exactly verifiable game-theory task.

The main result is positive: `Qwen/Qwen3.6-27B` improved from 87.60% exact-match accuracy at baseline to 99.60% after LoRA SFT on 5000 synthetic examples. The result held on an independent confirmation set and became sharper on a balanced stress set.

## Task

Each example presents a 2x2 payoff matrix:

- Player 1 chooses `U` or `D`.
- Player 2 chooses `L` or `R`.
- Each cell contains `(Player 1 payoff, Player 2 payoff)`.
- Payoffs are random integers from 0 to 9.

The target answer is the complete set of pure-strategy Nash equilibria. A cell is correct only when both players are best responding in that cell. Ties are handled exactly, so games may have zero, one, two, three, or four pure equilibria.

The benchmark intentionally excludes mixed strategies, dominance, welfare analysis, sequential games, auctions, public goods, Nim, and story problems.

## Data and Scoring

The canonical dataset uses seed `42` and contains:

- 5000 training examples
- 500 validation examples
- 500 canonical test examples

The generator avoids duplicate payoff matrices and exports both standard JSONL and chat-format JSONL. The chat format is used for supervised fine-tuning.

The scorer parses common model answer formats such as `(U, L)`, `U,L`, `(D, R)`, `none`, and `no pure Nash equilibrium`. It compares predicted equilibrium sets to the gold set while ignoring order.

Accuracy is exact-match accuracy: a prediction is correct only if it names exactly the same equilibrium set as the solver.

## Tinker Setup

The experiment used:

- Model: `Qwen/Qwen3.6-27B`
- Renderer: `qwen3_disable_thinking`
- Method: LoRA SFT
- LoRA rank: 16
- Epochs: 3
- Effective batch size: 32
- Learning rate: 1e-4
- Decoding: temperature 0, one sample

Three training sizes were evaluated: 250, 1000, and 5000 chat examples.

## Main Results

On the canonical 500-example test split:

| Run | Accuracy | 95% exact binomial CI | Correct | Incorrect | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 87.60% | 84.39%-90.36% | 438 | 62 | 0.00 pp |
| 250-example SFT | 53.40% | 48.92%-57.84% | 267 | 233 | -34.20 pp |
| 1000-example SFT | 91.60% | 88.82%-93.88% | 458 | 42 | +4.00 pp |
| 5000-example SFT | 99.60% | 98.56%-99.95% | 498 | 2 | +12.00 pp |

The 250-example run hurt performance, which is useful evidence that too little targeted data can distort the learned rule. The 1000-example run improved accuracy modestly, and the 5000-example run nearly solved the canonical split.

![Canonical accuracy](reports/figures/accuracy_main.svg)

## Confirmation and Stress Results

To check whether the result was split-specific, a fresh 1000-example confirmation set was generated with seed `20260505`. The same 5000-example checkpoint improved from 89.20% to 99.70%.

To probe the hardest cases more directly, a balanced 250-example stress set was generated with exactly 50 examples in each equilibrium-count bucket from 0 through 4. On this set, the baseline achieved 81.60%, while the 5000-example checkpoint achieved 100.00%.

![Confirmation and stress accuracy](reports/figures/accuracy_confirm_stress.svg)

The stress result shows the most important baseline weakness: zero-equilibrium games. The baseline solved only 16 of 50 zero-equilibrium stress examples, while the fine-tuned checkpoint solved all 50.

![Stress accuracy by equilibria](reports/figures/accuracy_by_equilibria.svg)

## Prompt Robustness

To test whether the checkpoint was overfitting the original prompt template, I added a 250-example robustness set with five prompt variants: original table, compact payoff pairs, JSON-like payoff object, minimal matrix, and answer-only instruction.

The 5000-example checkpoint improved from 64.00% baseline accuracy to 88.40%. The strongest gains were on `answer_only` and `standard_table`, while `compact_pairs` and `json_payoffs` remained weaker.

![Robustness by prompt variant](reports/figures/robustness_by_variant.svg)

This strengthens the result but also reveals the next frontier: the model is much better after fine-tuning, but not fully prompt-invariant.

## Adversarial Robustness Follow-Up

The finish-up milestone adds a targeted adversarial SFT pipeline for the weakest robustness variants. The final supplemental data contains 500 additional chat examples, weighted toward compact payoff pairs and JSON-like payoff objects while remaining balanced across 0-, 1-, 2-, 3-, and 4-equilibrium games.

The final run is `qwen36_27b_sft_5000_plus_prompt_adv500`: the original 5000-example training set plus the conservative adversarial prompt supplement, using the same `Qwen/Qwen3.6-27B` LoRA SFT settings. It met the acceptance target: prompt-robustness accuracy reached 98.80%, and canonical accuracy increased from 99.60% to 99.80%.

| Evaluation | Original 5000 SFT | Adversarial SFT | Delta |
| --- | ---: | ---: | ---: |
| canonical | 99.60% | 99.80% | +0.20 pp |
| confirmation | 99.70% | 99.90% | +0.20 pp |
| stress | 100.00% | 100.00% | +0.00 pp |
| robustness | 88.40% | 98.80% | +10.40 pp |

![Adversarial SFT comparison](reports/figures/adversarial_comparison.svg)

The public summary is `reports/adversarial_results.md`.

## Most Important Failure Mode

The base model often wants every 2x2 game to have at least one pure equilibrium. This creates many false positives on zero-equilibrium games, where the correct answer is the empty set.

After fine-tuning, this failure mode is mostly removed. The remaining canonical and confirmation failures in the best checkpoint are tie-heavy games where the model identifies true equilibria but over-predicts one extra cell that is not actually a mutual best response. The robustness run adds a second failure mode: compact or structured payoff presentations can still reduce accuracy. The adversarial follow-up substantially reduces that second failure mode without weakening the canonical result; the remaining robustness errors are concentrated in the JSON-like payoff format.

## Claim Boundaries

GT-Bench supports a narrow claim: targeted Tinker fine-tuning can substantially improve `Qwen/Qwen3.6-27B` on a synthetic, fully verifiable 2x2 pure-strategy Nash equilibrium task.

It does not show broad game-theory reasoning competence. It does not test mixed strategies, dominance, welfare analysis, sequential games, auctions, public goods, Nim, or natural-language story problems.

The value of the benchmark is control: the task is simple, exactly solvable, cheap to generate, and scored without subjective judgment.

## Reproduction

Core commands:

```bash
.venv/bin/python generate_dataset.py --train 5000 --val 500 --test 500 --seed 42 --out data/
.venv/bin/python make_sweep_splits.py --config bench_config.json
.venv/bin/python plot_results.py --summary reports/gt_bench_results.json --out-dir reports/figures
```

The full reproducibility recipe, including the complete summary command with all baseline, fine-tuned, confirmation, and stress reports, is in `REPRODUCIBILITY.md`.

## Next Steps

The next scientific step is a repeated-seed learning curve. A second benchmark should be added only as a separate task, rather than mixing new game-theory concepts into this controlled artifact.
