# GT-Bench Technical Report

## Summary

GT-Bench is a compact Tinker fine-tuning experiment for one formal reasoning task: given a 2x2 two-player normal-form payoff matrix, identify all pure-strategy Nash equilibria.

The repository also includes a broader task suite with exact solvers for mixed 2x2 equilibria, dominance, larger normal-form games, extensive-form backward induction, natural-language game descriptions, and repeated interaction, including fixed-policy simulation and best-response selection among policies. Those suite tasks are code-complete, exactly scored, split into deterministic train/validation/test files, covered by local smoke baselines, and now have a retention-aware multitask Tinker result. The headline canonical result remains the 2x2 pure-equilibrium task; the broader-suite result is reported as a bounded extension.

The project was built by Mert Gulsun, a UC Berkeley master's student and Thinking Machines Lab Tinker research grant recipient. The goal is to show whether targeted fine-tuning can measurably improve a model on a narrow, exactly verifiable game-theory task.

The main result is positive: `Qwen/Qwen3.6-27B` improved from 87.60% exact-match accuracy at baseline to 99.60% after LoRA SFT on 5000 synthetic examples. The result held on an independent confirmation set, became sharper on a balanced stress set, and remained stable in a three-seed training-data sweep. A retention-aware joint checkpoint then reached 91.67% on the broader suite while retaining 99.80% canonical accuracy and 99.60% prompt-robustness accuracy.

## Task

Each example presents a 2x2 payoff matrix:

- Player 1 chooses `U` or `D`.
- Player 2 chooses `L` or `R`.
- Each cell contains `(Player 1 payoff, Player 2 payoff)`.
- Payoffs are random integers from 0 to 9.

The target answer is the complete set of pure-strategy Nash equilibria. A cell is correct only when both players are best responding in that cell. Ties are handled exactly, so games may have zero, one, two, three, or four pure equilibria.

The canonical Tinker experiment intentionally excludes mixed strategies, dominance, welfare analysis, sequential games, auctions, public goods, Nim, and story problems. The separate suite generator covers several of those task families without changing the canonical result.

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

## Repeated-Seed Learning Curve

I repeated the training-size sweep across three independently generated training sets while keeping the canonical 500-example test set fixed. This tests whether the learning curve is stable under synthetic training-data seed variation.

| Train size | Mean accuracy | Seed SD | Mean delta vs baseline |
| ---: | ---: | ---: | ---: |
| 250 | 50.60% | 3.86 pp | -37.00 pp |
| 1000 | 84.67% | 12.53 pp | -2.93 pp |
| 5000 | 99.60% | 0.40 pp | +12.00 pp |

![Repeated-seed learning curve](reports/figures/seed_sweep_learning_curve.svg)

The result sharpens the original story. The 5000-example condition is stable across seeds, with individual accuracies of 99.60%, 99.20%, and 100.00%. The 1000-example condition is volatile: two seeds improve over baseline, but seed `2027` falls to 70.20%. The 250-example condition consistently underperforms baseline.

## Broader Suite And Retention-Aware Multitask Result

The broader suite test split has 300 examples, 50 from each task family. The suite-only run was useful but had poor canonical retention. The retention-aware sweep fixed that by selecting the best suite checkpoint among candidates satisfying canonical and prompt-robustness gates.

| Run | Suite accuracy | Correct | Incorrect |
| --- | ---: | ---: | ---: |
| base `Qwen/Qwen3.6-27B` | 20.00% | 60 | 240 |
| 2x2 SFT transfer | 48.33% | 145 | 155 |
| 2x2 + prompt-adversarial SFT transfer | 51.67% | 155 | 145 |
| suite SFT | 68.00% | 204 | 96 |
| joint base canonical+adv+suite | 84.67% | 254 | 46 |
| joint adv-state suite+retention | 89.33% | 268 | 32 |
| selected joint adv-state targeted+retention | 91.67% | 275 | 25 |
| joint base full targeted | 93.00% | 279 | 21 |

The selected checkpoint is `joint_adv_targeted_retention`. It scored 99.80% canonical, 100.00% confirmation, 100.00% stress, 99.60% robustness, and 91.67% suite. The higher-suite `joint_base_full_targeted` candidate reached 93.00% suite accuracy but missed the retention gate with 97.20% canonical and 94.40% robustness.

Across seeds `42`, `1009`, and `2027`, the selected recipe averaged 92.56% suite accuracy with 0.63 pp seed SD and 99.93% canonical accuracy with 0.09 pp seed SD. The largest per-family seed SDs were larger normal-form games (3.40 pp), mixed 2x2 games (2.49 pp), and dominance (1.63 pp).

The availability-selected external base models were weak under the exact parsers: `Qwen/Qwen3-8B` scored 2.60% canonical and 4.00% suite, while `Qwen/Qwen3-30B-A3B` scored 3.80% canonical and 6.33% suite.

## Most Important Failure Mode

The base model often wants every 2x2 game to have at least one pure equilibrium. This creates many false positives on zero-equilibrium games, where the correct answer is the empty set.

After fine-tuning, this failure mode is mostly removed. The selected joint checkpoint has one canonical false-negative in a tie-heavy three-equilibrium game. On the broader suite, its remaining failures are concentrated in dominance, larger normal-form games, and mixed 2x2 games. The robustness run adds a second failure mode: compact or structured payoff presentations can reduce accuracy. The adversarial follow-up substantially reduces that second failure mode without weakening the canonical result.

## Claim Boundaries

GT-Bench supports a bounded claim: targeted Tinker fine-tuning can substantially improve `Qwen/Qwen3.6-27B` on a synthetic, fully verifiable 2x2 pure-strategy Nash equilibrium task, and retention-aware multitask fine-tuning can improve a broader exact suite without erasing that canonical skill.

It does not show broad game-theory reasoning competence. The broader suite now tests mixed strategies, dominance, larger normal-form games, sequential games, natural-language descriptions, and repeated interaction, but it is still synthetic and exact-format.

The value of the benchmark is control: the task is simple, exactly solvable, cheap to generate, and scored without subjective judgment.

## Reproduction

Core commands:

```bash
.venv/bin/python generate_dataset.py --train 5000 --val 500 --test 500 --seed 42 --out data/
.venv/bin/python make_sweep_splits.py --config bench_config.json
.venv/bin/python make_multitask_training.py
.venv/bin/python summarize_multitask_results.py
.venv/bin/python plot_results.py --summary reports/gt_bench_results.json --out-dir reports/figures
```

The full reproducibility recipe, including the complete summary command with all baseline, fine-tuned, confirmation, and stress reports, is in `REPRODUCIBILITY.md`.

## Remaining Gaps

The remaining scientific gaps are outside the current artifact: expand beyond exact synthetic answer formats, add economically richer games such as auctions and bargaining, and test whether improvements transfer to natural analysis rather than fixed final-answer extraction.
