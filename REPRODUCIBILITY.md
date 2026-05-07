# Reproducibility

This file records the exact GT-Bench demo setup.

## Environment

- Python: 3.11+ required; local run used Python 3.14.4
- Core dependency: standard library
- Test dependency: `pytest`
- Tinker dependencies: `tinker==0.18.2`, `tinker-cookbook==0.3.0`
- Model: `Qwen/Qwen3.6-27B`
- Renderer: `qwen3_disable_thinking`
- Method: LoRA SFT

## Canonical Data

Generate the main data:

```bash
.venv/bin/python generate_dataset.py --train 5000 --val 500 --test 500 --seed 42 --out data/
```

Canonical test hash:

```text
a9af74346a7b67cc13952fc858570dee6d03e4248b04e830c42f8c1bb935d219  data/test.jsonl
```

Training file hash:

```text
8e2201a88022dfb6f37a71f41582d7c1053548149d3757205cf840bc053ac18b  data/train_chat.jsonl
```

Create sweep files:

```bash
.venv/bin/python make_sweep_splits.py --config bench_config.json
```

## Confirmation Data

Generate the independent confirmation set:

```bash
mkdir -p data/confirm
.venv/bin/python generate_dataset.py --train 0 --val 0 --test 1000 --seed 20260505 --out data/confirm
mv data/confirm/test.jsonl data/confirm/confirm_seed20260505.jsonl
mv data/confirm/test_chat.jsonl data/confirm/confirm_seed20260505_chat.jsonl
rm -f data/confirm/train.jsonl data/confirm/train_chat.jsonl data/confirm/val.jsonl data/confirm/val_chat.jsonl
```

Confirmation hash:

```text
d9874764ce086f23065568ba5066745e0a879955a924e8a257d2f3b1f82a0cd8  data/confirm/confirm_seed20260505.jsonl
```

## Balanced Stress Data

Generate the balanced stress set:

```bash
.venv/bin/python generate_stress_set.py \
  --per-count 50 \
  --seed 314159 \
  --out data/stress/tie_stress_seed314159.jsonl \
  --chat-out data/stress/tie_stress_seed314159_chat.jsonl
```

This creates 250 examples: 50 each with 0, 1, 2, 3, and 4 pure-strategy equilibria.

Stress set hashes:

```text
9c298c756940bb57189f114da491f8b56b57dd9bedddefe82e61c82669f94340  data/stress/tie_stress_seed314159.jsonl
d5b397b1078794609016f1f6fd741f0ba3e1df9ca95eb0c356303a88f596e953  data/stress/tie_stress_seed314159_chat.jsonl
```

## Prompt-Robustness Data

Generate the prompt-robustness set:

```bash
.venv/bin/python generate_robustness_set.py \
  --per-bucket 10 \
  --seed 271828 \
  --out data/robust/robust_seed271828.jsonl \
  --chat-out data/robust/robust_seed271828_chat.jsonl
```

This creates 250 examples: 5 prompt variants x 5 equilibrium-count buckets x 10 examples.

Robustness set hashes:

```text
ffc6c26df85c78df702d4eef910b835025a5b2a8344a9ef1d14e1ab7e78405e8  data/robust/robust_seed271828.jsonl
69fd257b7cda77dac18de86fce775b401283d4e676c33d35a458e7b991ce6906  data/robust/robust_seed271828_chat.jsonl
```

## Adversarial Prompt Data

Generate the adversarial prompt supplement:

```bash
.venv/bin/python generate_adversarial_training.py
```

This creates 1000 supplemental examples and a 6000-row combined chat file:

```text
data/adversarial/prompt_adv_seed161803.jsonl
data/adversarial/prompt_adv_seed161803_chat.jsonl
data/adversarial/train_6000_prompt_adv_chat.jsonl
```

The supplement avoids matrices already present in `data/train.jsonl`, emphasizes `compact_pairs` and `json_payoffs`, and balances every prompt variant across 0, 1, 2, 3, and 4 equilibria.

## Training

The main sweep used:

- train sizes: 250, 1000, 5000
- LoRA rank: 16
- epochs: 3
- effective batch size: 32
- learning rate: 1e-4
- decoding temperature: 0.0
- max generated tokens: 768

Run preflight:

```bash
.venv/bin/python run_tinker_preflight.py --config bench_config.json
```

Run the best 5000-example SFT:

```bash
.venv/bin/python run_tinker_sft.py \
  --config bench_config.json \
  --train-chat data/sweeps/train_5000_chat.jsonl \
  --run-name qwen36_27b_sft_5000 \
  --out-manifest runs/qwen36_27b_sft_5000.json
```

Run the adversarial follow-up SFT:

```bash
.venv/bin/python run_tinker_sft.py \
  --config bench_config.json \
  --train-chat data/adversarial/train_6000_prompt_adv_chat.jsonl \
  --run-name qwen36_27b_sft_5000_plus_prompt_adv \
  --out-manifest runs/qwen36_27b_sft_5000_plus_prompt_adv.json
```

## Evaluation

Score predictions with:

```bash
.venv/bin/python score_predictions.py --gold data/test.jsonl --pred predictions/baseline_qwen36_27b.jsonl --out reports/baseline_qwen36_27b_report.json
.venv/bin/python score_predictions.py --gold data/test.jsonl --pred predictions/qwen36_27b_sft_5000.jsonl --out reports/qwen36_27b_sft_5000_report.json
```

Run and score the stress evaluations with:

```bash
.venv/bin/python run_tinker_predict.py \
  --config bench_config.json \
  --gold data/stress/tie_stress_seed314159.jsonl \
  --out predictions/stress_baseline_qwen36_27b_seed314159.jsonl

.venv/bin/python score_predictions.py \
  --gold data/stress/tie_stress_seed314159.jsonl \
  --pred predictions/stress_baseline_qwen36_27b_seed314159.jsonl \
  --out reports/stress_baseline_qwen36_27b_seed314159_report.json

.venv/bin/python run_tinker_predict.py \
  --config bench_config.json \
  --gold data/stress/tie_stress_seed314159.jsonl \
  --model-path "tinker://..." \
  --out predictions/stress_qwen36_27b_sft_5000_seed314159.jsonl

.venv/bin/python score_predictions.py \
  --gold data/stress/tie_stress_seed314159.jsonl \
  --pred predictions/stress_qwen36_27b_sft_5000_seed314159.jsonl \
  --out reports/stress_qwen36_27b_sft_5000_seed314159_report.json
```

Generate the public summary:

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

The summary includes exact-match accuracy, percentage-point deltas, per-equilibrium-count deltas in JSON, and 95% exact binomial confidence intervals.

Generate the public figures:

```bash
.venv/bin/python plot_results.py \
  --summary reports/gt_bench_results.json \
  --robustness reports/robustness_results.json \
  --out-dir reports/figures
```

Run and summarize the robustness evaluation with:

```bash
.venv/bin/python run_tinker_predict.py \
  --config bench_config.json \
  --gold data/robust/robust_seed271828.jsonl \
  --out predictions/robust_baseline_qwen36_27b_seed271828.jsonl

.venv/bin/python score_robustness.py \
  --gold data/robust/robust_seed271828.jsonl \
  --pred predictions/robust_baseline_qwen36_27b_seed271828.jsonl \
  --out reports/robust_baseline_qwen36_27b_seed271828_report.json

.venv/bin/python run_tinker_predict.py \
  --config bench_config.json \
  --gold data/robust/robust_seed271828.jsonl \
  --model-path "tinker://..." \
  --out predictions/robust_qwen36_27b_sft_5000_seed271828.jsonl

.venv/bin/python score_robustness.py \
  --gold data/robust/robust_seed271828.jsonl \
  --pred predictions/robust_qwen36_27b_sft_5000_seed271828.jsonl \
  --out reports/robust_qwen36_27b_sft_5000_seed271828_report.json

.venv/bin/python summarize_robustness.py \
  --gold data/robust/robust_seed271828.jsonl \
  --baseline reports/robust_baseline_qwen36_27b_seed271828_report.json \
  --finetuned reports/robust_qwen36_27b_sft_5000_seed271828_report.json \
  --out-json reports/robustness_results.json \
  --out-md reports/robustness_results.md
```

After scoring the adversarial checkpoint on the canonical, confirmation, stress, and robustness sets, regenerate the adversarial public tracker:

```bash
.venv/bin/python summarize_adversarial.py \
  --out-json reports/adversarial_results.json \
  --out-md reports/adversarial_results.md

.venv/bin/python plot_results.py \
  --summary reports/gt_bench_results.json \
  --robustness reports/robustness_results.json \
  --adversarial reports/adversarial_results.json \
  --out-dir reports/figures
```
