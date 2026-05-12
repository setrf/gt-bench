# Release Notes

## v0.4-retention-aware-multitask - 2026-05-12

- Adds a broader exactly scored GT-Bench suite with `mixed_2x2`, `dominance`, `large_normal_form`, `extensive_form`, `natural_language`, and `repeated_interaction` task families.
- Adds split generation, v1 suite schema metadata, difficulty buckets, `generate_benchmark_suite.py`, `score_suite.py`, `run_suite_baselines.py`, suite solver tests, and small tracked suite samples under `examples/`.
- Adds local deterministic suite smoke baselines, completed Tinker suite model rows, canonical-retention reporting, and `reports/figures/suite_smoke_accuracy.svg`.
- Adds retention-aware multitask training data builders, Tinker continuation support, model-availability reporting, multitask result aggregation, failure diagnostics, and public summaries under `reports/multitask_results.*`.
- Adds an explicit experiment coverage matrix documenting completed candidate, seed, and external-model combinations plus the reason conditional follow-up SFT was not required.
- Selects `joint_adv_targeted_retention` by the retention rule: 91.67% suite, 99.80% canonical, and 99.60% robustness.
- Adds a three-seed selected-recipe repeat: 92.56% mean suite accuracy with 0.63 pp seed SD, and 99.93% mean canonical accuracy with 0.09 pp seed SD.
- Adds external base-model comparisons for `Qwen/Qwen3-8B` and `Qwen/Qwen3-30B-A3B` selected from Tinker model availability.
- Expands `paper/gt_bench_paper.tex` into an arXiv-ready draft covering canonical, adversarial, broader-suite, and retention-aware multitask results.
- Adds related work, leakage controls, simple baselines, training-time notes, parser details, a concrete failure example, exact adversarial supplement mix, reproducibility appendix, and code/data/ethics notes.
- Adds arXiv-safe PNG paper figures under `paper/figures/`, including broader-suite, Pareto, and external-base figures.
- Documents the paper build and source-zip workflow while keeping generated paper outputs ignored under `paper/build/`.

## v0.3-repeated-seed-learning-curve

- Adds a repeated-seed SFT learning curve across training-data seeds `42`, `1009`, and `2027`.
- Keeps evaluation fixed on the canonical 500-example test set to isolate training-data seed variation.
- Shows 5000-example SFT is stable across seeds: 99.60% mean accuracy with 0.40 percentage-point seed SD.
- Shows 1000-example SFT is volatile: mean accuracy 84.67% with 12.53 percentage-point seed SD.
- Adds `make_repeated_seed_splits.py`, `summarize_seed_sweep.py`, `reports/seed_sweep_results.*`, and `reports/figures/seed_sweep_learning_curve.svg`.
- Adds bounded concurrent Tinker inference via `run_tinker_predict.py --concurrency`.

## v0.2-adversarial-robustness

- Adds the adversarial prompt SFT result for the robustness gap found in `v0.1.3`.
- Generates a deterministic 500-example supplemental training set weighted toward `compact_pairs` and `json_payoffs`.
- Improves robustness accuracy from 88.40% to 98.80% versus the original 5000-example SFT checkpoint, while canonical accuracy increases from 99.60% to 99.80%.
- Improves `compact_pairs` from 78.00% to 100.00% and `json_payoffs` from 68.00% to 94.00%.
- Adds `reports/adversarial_results.*` and `reports/figures/adversarial_comparison.svg`.
- Adds a Makefile and GitHub Actions CI for tests, secret scanning, and public artifact regeneration.

## v0.1.3-robustness-eval

- Adds a 250-example prompt-robustness evaluation set with five prompt variants.
- Evaluates baseline and the 5000-example SFT checkpoint without additional training.
- Improves robustness accuracy from 64.00% baseline to 88.40% after fine-tuning.
- Adds prompt-robustness documentation in `reports/robustness_results.*` and a prompt-variant SVG figure.

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
