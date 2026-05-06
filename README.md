# GT-Bench: A Minimal Game-Theory Fine-Tuning Benchmark

## Project overview

GT-Bench is a minimal Tinker fine-tuning benchmark for strategic reasoning. It generates small, fully verifiable 2x2 normal-form games and asks a model to find all pure-strategy Nash equilibria with a brief explanation.

The repository is intentionally compact: one task, one exact solver, one dataset generator, one scorer, and focused tests.

For the consolidated research narrative, see `TECHNICAL_REPORT.md`.

## My context

This project is by Mert Gulsun, a UC Berkeley master's student and Thinking Machines Lab Tinker research grant recipient.

The goal is to demonstrate measurable improvement from targeted fine-tuning on formal game-theory reasoning using a 12-month, $5,000 Tinker research credit allocation.

## Task

Each example is a 2x2 two-player normal-form payoff matrix.

- Player 1 chooses `U` or `D`.
- Player 2 chooses `L` or `R`.
- Each cell contains `(Player 1 payoff, Player 2 payoff)`.

The model must find all pure-strategy Nash equilibria. A profile is a pure Nash equilibrium when both players are best responding at that cell. Ties are handled exactly, so a game may have zero, one, two, three, or four pure equilibria.

This benchmark does not include mixed strategies, dominance, welfare analysis, sequential games, auctions, public goods, Nim, or story problems.

## Why this is useful

The task is narrow, synthetic, and fully verifiable. That makes it useful for testing whether targeted fine-tuning improves a specific reasoning skill instead of relying on subjective grading.

Because examples are easy to generate at scale and evaluate exactly, GT-Bench can support a clean before-and-after comparison between a base model and a fine-tuned model.

## How to generate data

Install the test dependency if needed:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Generate train, validation, and test splits:

```bash
.venv/bin/python generate_dataset.py --train 5000 --val 500 --test 500 --seed 42 --out data/
```

This writes:

- `data/train.jsonl`
- `data/val.jsonl`
- `data/test.jsonl`
- `data/train_chat.jsonl`
- `data/val_chat.jsonl`
- `data/test_chat.jsonl`

Small sample files are included in `examples/`.

Generate a balanced stress set with equal numbers of 0-, 1-, 2-, 3-, and 4-equilibrium games:

```bash
.venv/bin/python generate_stress_set.py \
  --per-count 50 \
  --seed 314159 \
  --out data/stress/tie_stress_seed314159.jsonl \
  --chat-out data/stress/tie_stress_seed314159_chat.jsonl
```

## How to fine-tune

Use `data/train_chat.jsonl` as the fine-tuning file on Tinker. Each row contains a user message with the matrix prompt and an assistant message with:

- a final answer listing the pure-strategy Nash equilibria
- concise reasoning based on exact best responses

Use `data/val_chat.jsonl` as a held-out validation file if the fine-tuning workflow supports it.

For the Qwen3.6-27B Tinker experiment, install the optional Tinker dependencies:

```bash
.venv/bin/python -m pip install -r requirements-tinker.txt
```

Export your Tinker key locally. Do not put it in tracked files, shell scripts, notebooks, reports, or README examples:

```bash
export TINKER_API_KEY="..."
```

Validate access to the exact model configured in `bench_config.json`:

```bash
.venv/bin/python run_tinker_preflight.py --config bench_config.json
```

Create the training-size sweep files:

```bash
.venv/bin/python make_sweep_splits.py --config bench_config.json
```

This writes:

- `data/sweeps/train_0250_chat.jsonl`
- `data/sweeps/train_1000_chat.jsonl`
- `data/sweeps/train_5000_chat.jsonl`

Run the baseline on `Qwen/Qwen3.6-27B`:

```bash
.venv/bin/python run_tinker_predict.py \
  --config bench_config.json \
  --gold data/test.jsonl \
  --out predictions/baseline_qwen36_27b.jsonl
```

`bench_config.json` uses deterministic decoding with `max_tokens` set high enough to let the base model finish its reasoning and final answer. You can override this with `--max-tokens` for smoke tests.

Run the three LoRA SFT jobs:

```bash
.venv/bin/python run_tinker_sft.py \
  --config bench_config.json \
  --train-chat data/sweeps/train_0250_chat.jsonl \
  --run-name qwen36_27b_sft_0250 \
  --out-manifest runs/qwen36_27b_sft_0250.json

.venv/bin/python run_tinker_sft.py \
  --config bench_config.json \
  --train-chat data/sweeps/train_1000_chat.jsonl \
  --run-name qwen36_27b_sft_1000 \
  --out-manifest runs/qwen36_27b_sft_1000.json

.venv/bin/python run_tinker_sft.py \
  --config bench_config.json \
  --train-chat data/sweeps/train_5000_chat.jsonl \
  --run-name qwen36_27b_sft_5000 \
  --out-manifest runs/qwen36_27b_sft_5000.json
```

Each manifest records the sampler checkpoint path. Use that path to evaluate the fine-tuned run:

```bash
.venv/bin/python run_tinker_predict.py \
  --config bench_config.json \
  --gold data/test.jsonl \
  --model-path "tinker://..." \
  --out predictions/qwen36_27b_sft_0250.jsonl
```

## How to evaluate

Run the base model on `data/test.jsonl` and save predictions as JSONL:

```json
{"id": "example_000001", "prediction": "The pure-strategy Nash equilibria are (U, L) and (D, R)."}
```

Score the baseline predictions:

```bash
.venv/bin/python score_predictions.py --gold data/test.jsonl --pred predictions_baseline.jsonl --out reports/baseline_report.json
```

Run the fine-tuned model on the same `data/test.jsonl`, save predictions to `predictions_finetuned.jsonl`, and score them:

```bash
.venv/bin/python score_predictions.py --gold data/test.jsonl --pred predictions_finetuned.jsonl --out reports/finetuned_report.json
```

Compare exact-match accuracy between `reports/baseline_report.json` and `reports/finetuned_report.json`.

For the Qwen3.6-27B sweep, score each prediction file:

```bash
.venv/bin/python score_predictions.py \
  --gold data/test.jsonl \
  --pred predictions/baseline_qwen36_27b.jsonl \
  --out reports/baseline_qwen36_27b_report.json

.venv/bin/python score_predictions.py \
  --gold data/test.jsonl \
  --pred predictions/qwen36_27b_sft_0250.jsonl \
  --out reports/qwen36_27b_sft_0250_report.json

.venv/bin/python score_predictions.py \
  --gold data/test.jsonl \
  --pred predictions/qwen36_27b_sft_1000.jsonl \
  --out reports/qwen36_27b_sft_1000_report.json

.venv/bin/python score_predictions.py \
  --gold data/test.jsonl \
  --pred predictions/qwen36_27b_sft_5000.jsonl \
  --out reports/qwen36_27b_sft_5000_report.json
```

Then summarize the reports:

```bash
.venv/bin/python summarize_results.py \
  --config bench_config.json \
  --baseline reports/baseline_qwen36_27b_report.json \
  --run qwen36_27b_sft_0250=reports/qwen36_27b_sft_0250_report.json \
  --run qwen36_27b_sft_1000=reports/qwen36_27b_sft_1000_report.json \
  --run qwen36_27b_sft_5000=reports/qwen36_27b_sft_5000_report.json \
  --confirmation-gold data/confirm/confirm_seed20260505.jsonl \
  --confirmation-baseline reports/confirm_baseline_qwen36_27b_seed20260505_report.json \
  --confirmation-run qwen36_27b_sft_5000=reports/confirm_qwen36_27b_sft_5000_seed20260505_report.json \
  --stress-gold data/stress/tie_stress_seed314159.jsonl \
  --stress-baseline reports/stress_baseline_qwen36_27b_seed314159_report.json \
  --stress-run qwen36_27b_sft_5000=reports/stress_qwen36_27b_sft_5000_seed314159_report.json \
  --out-json reports/gt_bench_results.json \
  --out-md reports/gt_bench_results.md
```

Generate static SVG figures from the public summary:

```bash
.venv/bin/python plot_results.py --summary reports/gt_bench_results.json --out-dir reports/figures
```

## Expected result

The expected demonstration is an increase in exact-match accuracy on held-out 2x2 Nash equilibrium problems after fine-tuning.

## Current Qwen3.6-27B result

On the canonical 500-example held-out test split, `Qwen/Qwen3.6-27B` improved from 87.60% exact-match accuracy at baseline to 99.60% after LoRA SFT on 5000 synthetic examples.

| Run | Accuracy | Correct | Incorrect | Delta vs baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 87.60% | 438 | 62 | 0.00 pp |
| 250-example SFT | 53.40% | 267 | 233 | -34.20 pp |
| 1000-example SFT | 91.60% | 458 | 42 | +4.00 pp |
| 5000-example SFT | 99.60% | 498 | 2 | +12.00 pp |

On an independent 1000-example confirmation set generated with seed `20260505`, the same 5000-example fine-tuned checkpoint improved from 89.20% baseline accuracy to 99.70%.

On a balanced 250-example stress set with 50 examples in each equilibrium-count bucket from 0 through 4, the same checkpoint improved from 81.60% baseline accuracy to 100.00%.

The full report is in `reports/gt_bench_results.md`. For the research narrative and reproducibility details, see `RESULTS.md`, `REPRODUCIBILITY.md`, and `FAILURE_ANALYSIS.md`.

On a 250-example prompt-robustness set, the same checkpoint improved from 64.00% baseline accuracy to 88.40%. See `ROBUSTNESS.md` for the prompt-variant breakdown.

## Result figures

![Canonical test accuracy](reports/figures/accuracy_main.svg)

![Confirmation and stress accuracy](reports/figures/accuracy_confirm_stress.svg)

![Stress accuracy by number of equilibria](reports/figures/accuracy_by_equilibria.svg)

![Robustness accuracy by prompt variant](reports/figures/robustness_by_variant.svg)

## Limitations

GT-Bench is deliberately narrow. It uses synthetic data, covers pure equilibria only, and does not prove broad game-theory reasoning improvement.

It is best understood as a controlled fine-tuning benchmark for one formal reasoning task, not as a general game-theory benchmark.

## Testing

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python check_no_secrets.py
```

The tests cover a Prisoner's Dilemma style one-equilibrium game, a coordination game with two equilibria, a matching pennies style game with no pure equilibrium, a tie case with multiple best responses, and common prediction parser formats.

`check_no_secrets.py` scans tracked git files for Tinker API key markers before committing.
