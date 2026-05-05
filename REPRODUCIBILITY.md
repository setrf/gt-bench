# Reproducibility

This file records the exact GT-Bench v0.1 demo setup.

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

## Evaluation

Score predictions with:

```bash
.venv/bin/python score_predictions.py --gold data/test.jsonl --pred predictions/baseline_qwen36_27b.jsonl --out reports/baseline_qwen36_27b_report.json
.venv/bin/python score_predictions.py --gold data/test.jsonl --pred predictions/qwen36_27b_sft_5000.jsonl --out reports/qwen36_27b_sft_5000_report.json
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
  --out-json reports/gt_bench_results.json \
  --out-md reports/gt_bench_results.md
```
