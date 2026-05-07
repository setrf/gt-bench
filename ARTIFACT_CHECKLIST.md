# Artifact Checklist

Use this checklist before publishing a GT-Bench release or sharing the repo as a research artifact.

## Code and Tests

- Run `.venv/bin/python -m pytest -q`.
- Run `.venv/bin/python check_no_secrets.py`.
- Run `make public-artifacts` or `.venv/bin/python summarize_adversarial.py` plus `.venv/bin/python plot_results.py`.
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
  --out-dir reports/figures
```

## Public-Release Safety

- Do not commit `.env`, `predictions/`, `runs/`, raw scorer reports, or generated JSONL data.
- Do not commit private Tinker sampler paths in public prose.
- Confirm no tracked file contains Tinker secret markers:

```bash
.venv/bin/python check_no_secrets.py
```

## Public Narrative

- Confirm `README.md` links to `TECHNICAL_REPORT.md`, `RESULTS.md`, `REPRODUCIBILITY.md`, `FAILURE_ANALYSIS.md`, and `ROBUSTNESS.md`.
- Confirm `reports/adversarial_results.md` accurately says whether the adversarial run is pending or complete.
- Confirm `reports/figures/*.svg` renders on GitHub.
- Confirm the headline claim remains narrow: targeted fine-tuning improves one fully verifiable 2x2 pure-equilibrium task.
