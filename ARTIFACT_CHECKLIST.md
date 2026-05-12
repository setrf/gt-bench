# Artifact Checklist

Use this checklist before publishing a GT-Bench release or sharing the repo as a research artifact.

## Code and Tests

- Run `.venv/bin/python -m pytest -q`.
- Run `.venv/bin/python check_no_secrets.py`.
- Run `make public-artifacts` or the individual summary scripts plus `.venv/bin/python plot_results.py`.
- Compile `paper/gt_bench_paper.tex` and, if preparing arXiv, test the source zip from a clean temporary directory.
- Run `git diff --check`.
- Confirm `git status -sb` contains only intended tracked changes.

## Data and Reports

- Regenerate canonical data when changing the generator:

```bash
.venv/bin/python generate_dataset.py --train 5000 --val 500 --test 500 --seed 42 --out data/
```

- Regenerate the balanced stress set when changing stress generation:

```bash
.venv/bin/python generate_stress_set.py \
  --per-count 50 \
  --seed 314159 \
  --out data/stress/tie_stress_seed314159.jsonl \
  --chat-out data/stress/tie_stress_seed314159_chat.jsonl
```

- Regenerate the prompt-robustness set when changing robustness generation:

```bash
.venv/bin/python generate_robustness_set.py \
  --per-bucket 10 \
  --seed 271828 \
  --out data/robust/robust_seed271828.jsonl \
  --chat-out data/robust/robust_seed271828_chat.jsonl
```

- Regenerate the adversarial prompt supplement when changing adversarial generation:

```bash
.venv/bin/python generate_adversarial_training.py
```

- Regenerate repeated-seed train splits when changing repeated-seed generation:

```bash
.venv/bin/python make_repeated_seed_splits.py --seed 1009 --seed 2027
```

- Regenerate the broader suite and local suite baselines when changing suite generation/scoring:

```bash
.venv/bin/python generate_benchmark_suite.py \
  --train-per-family 200 \
  --val-per-family 10 \
  --test-per-family 50 \
  --seed 20260511 \
  --out-dir data/suite

.venv/bin/python run_suite_baselines.py \
  --gold data/suite/test.jsonl \
  --train data/suite/train.jsonl \
  --pred-dir predictions/suite_baselines \
  --out-json reports/suite_results.json \
  --out-md reports/suite_results.md \
  --figure reports/figures/suite_smoke_accuracy.svg
```

- Regenerate multitask training manifests and model-availability reports when changing retention-aware recipes or external comparison logic:

```bash
.venv/bin/python make_multitask_training.py
.venv/bin/python select_tinker_models.py
.venv/bin/python summarize_multitask_results.py
```

- Regenerate the public summary after scoring:

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

- Regenerate the public robustness summary after robustness scoring:

```bash
.venv/bin/python summarize_robustness.py \
  --gold data/robust/robust_seed271828.jsonl \
  --baseline reports/robust_baseline_qwen36_27b_seed271828_report.json \
  --finetuned reports/robust_qwen36_27b_sft_5000_seed271828_report.json \
  --out-json reports/robustness_results.json \
  --out-md reports/robustness_results.md
```

- Regenerate figures:

```bash
.venv/bin/python plot_results.py \
  --summary reports/gt_bench_results.json \
  --robustness reports/robustness_results.json \
  --adversarial reports/adversarial_results.json \
  --seed-sweep reports/seed_sweep_results.json \
  --out-dir reports/figures
```

- Regenerate paper PNGs from updated public figures or chart data before compiling `paper/gt_bench_paper.tex`.

## Public-Release Safety

- Do not commit `.env`, `predictions/`, `runs/`, raw scorer reports, or generated JSONL data.
- Do not commit `paper/build/`, generated PDFs, or arXiv zip files.
- Do not commit private Tinker sampler paths in public prose.
- Confirm no tracked file contains Tinker secret markers:

```bash
.venv/bin/python check_no_secrets.py
```

## Public Narrative

- Confirm `README.md` links to `TECHNICAL_REPORT.md`, `RESULTS.md`, `REPRODUCIBILITY.md`, `FAILURE_ANALYSIS.md`, `ROBUSTNESS.md`, and `paper/gt_bench_paper.tex`.
- Confirm `reports/adversarial_results.md` accurately says whether the adversarial run is pending or complete.
- Confirm `reports/suite_results.md` accurately reports completed broader-suite model rows and canonical-retention status for the exact public suite hash.
- Confirm `reports/multitask_results.md` accurately reports selected-checkpoint rationale, three-seed statistics, external base comparisons, and failure diagnostics.
- Confirm `reports/figures/*.svg` renders on GitHub.
- Confirm `paper/figures/*.png` match the current public SVG figures before rebuilding the paper.
- Confirm the headline claim remains bounded: targeted fine-tuning improves the canonical 2x2 pure-equilibrium task, and the selected retention-aware multitask checkpoint broadens exact suite coverage without claiming general game-theory competence.
