# Release Notes

## v0.2-adversarial-robustness

- Adds the adversarial prompt SFT pipeline for the robustness gap found in `v0.1.3`.
- Generates a deterministic 1000-example supplemental training set weighted toward `compact_pairs` and `json_payoffs`.
- Adds `reports/adversarial_results.*` and `reports/figures/adversarial_comparison.svg`.
- Adds a Makefile and GitHub Actions CI for tests, secret scanning, and public artifact regeneration.
- Release should be finalized after the bounded Tinker SFT/evaluation run meets the documented acceptance criteria.

## v0.1.3-robustness-eval

- Adds a 250-example prompt-robustness evaluation set with five prompt variants.
- Evaluates baseline and the 5000-example SFT checkpoint without additional training.
- Improves robustness accuracy from 64.00% baseline to 88.40% after fine-tuning.
- Adds `ROBUSTNESS.md`, `reports/robustness_results.*`, and a prompt-variant SVG figure.

## v0.1.2-artifact-polish

- Adds static SVG result figures generated from `reports/gt_bench_results.json`.
- Adds exact binomial confidence intervals and per-bucket deltas to the public summary.
- Adds a technical report and artifact checklist for easier public review and reproduction.

## v0.1.1-stress-eval

- Adds a balanced stress-set generator for equilibrium-count buckets.
- Evaluates the baseline and best 5000-example SFT checkpoint on a 250-example stress set.
- Improves stress-set exact-match accuracy from 81.60% baseline to 100.00% after fine-tuning.

## v0.1-tinker-demo

GT-Bench v0.1 is a minimal, reproducible Tinker fine-tuning demo for one task: solve 2x2 pure-strategy Nash equilibria.

Highlights:

- Generates deterministic synthetic 2x2 normal-form games.
- Exports standard and chat JSONL data.
- Scores model predictions exactly.
- Runs a Qwen3.6-27B Tinker LoRA SFT training-size sweep.
- Improves canonical exact-match accuracy from 87.60% to 99.60%.
- Confirms on a fresh 1000-example set, improving from 89.20% to 99.70%.

Known limitation:

- The benchmark is narrow and synthetic; it does not demonstrate broad game-theory reasoning.
