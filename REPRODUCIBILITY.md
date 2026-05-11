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

## Broader Suite Data

The headline Qwen3.6-27B fine-tuning result is still the canonical 2x2 pure-equilibrium experiment. The repository also includes a broader suite with exact solvers, task-family scoring, local smoke baselines, completed Tinker model rows for the public split, and a canonical-retention check for the suite-specific checkpoint.

Generate one suite file:

```bash
.venv/bin/python generate_benchmark_suite.py \
  --per-family 50 \
  --seed 20260511 \
  --out data/suite/gt_bench_suite.jsonl \
  --chat-out data/suite/gt_bench_suite_chat.jsonl
```

This creates 300 examples: 50 each for `mixed_2x2`, `dominance`, `large_normal_form`, `extensive_form`, `natural_language`, and `repeated_interaction`.

Generate the public train/validation/test suite splits:

```bash
.venv/bin/python generate_benchmark_suite.py \
  --train-per-family 200 \
  --val-per-family 10 \
  --test-per-family 50 \
  --seed 20260511 \
  --out-dir data/suite
```

This creates 1200 train examples, 60 validation examples, and 300 test examples.

Suite split hashes:

```text
6b59d21ee17ab33eed9f0c14cb266953fbeb7dea65e48a1baae116bd63166067  data/suite/train.jsonl
e87e71a870e90fc6b7f6ec74c38a49d14614a33243e01eb3f621dfc60f6e3628  data/suite/val.jsonl
653aa5e1fd7bbcb2ff57e98c04cfc745a602f274c86dc8284bf875e650d1deee  data/suite/test.jsonl
```

Score suite predictions with:

```bash
.venv/bin/python score_suite.py \
  --gold data/suite/test.jsonl \
  --pred predictions/suite_predictions.jsonl \
  --out reports/suite_report.json
```

Generated suite JSONL files and raw suite reports are ignored by git. The tracked sample files are `examples/sample_suite.jsonl` and `examples/sample_suite_chat.jsonl`.

Regenerate local suite smoke baselines and the public summary:

```bash
.venv/bin/python run_suite_baselines.py \
  --gold data/suite/test.jsonl \
  --train data/suite/train.jsonl \
  --pred-dir predictions/suite_baselines \
  --out-json reports/suite_results.json \
  --out-md reports/suite_results.md \
  --figure reports/figures/suite_smoke_accuracy.svg
```

The public suite summary includes the completed model rows for the exact public suite hash above. If raw suite report files are absent, `run_suite_baselines.py` preserves the tracked model rows instead of requiring a Tinker key for local artifact checks.

The suite-specific SFT run used the generated suite chat split:

```bash
.venv/bin/python run_tinker_sft.py \
  --config bench_config.json \
  --train-chat data/suite/train_chat.jsonl \
  --val-chat data/suite/val_chat.jsonl \
  --run-name qwen36_27b_suite_sft_1200 \
  --out-manifest runs/qwen36_27b_suite_sft_1200.json
```

Evaluate suite checkpoints with `run_tinker_predict.py` and score with `score_suite.py`. For fine-tuned checkpoints, pass the `sampler_path` recorded in the relevant manifest as `--model-path`:

```bash
.venv/bin/python run_tinker_predict.py \
  --config bench_config.json \
  --gold data/suite/test.jsonl \
  --out predictions/suite_base_qwen36_27b.jsonl \
  --max-tokens 512 \
  --concurrency 16

.venv/bin/python score_suite.py \
  --gold data/suite/test.jsonl \
  --pred predictions/suite_base_qwen36_27b.jsonl \
  --out reports/suite_base_qwen36_27b_report.json
```

Canonical retention for the suite checkpoint is scored with `score_predictions.py` on `data/test.jsonl`.

The public suite pilot results on `data/suite/test.jsonl` are:

| Run | Accuracy | Correct | Incorrect |
| --- | ---: | ---: | ---: |
| base `Qwen/Qwen3.6-27B` | 20.00% | 60 | 240 |
| 2x2 SFT transfer | 48.33% | 145 | 155 |
| 2x2 + prompt-adversarial SFT transfer | 51.67% | 155 | 145 |
| suite SFT | 68.00% | 204 | 96 |

The suite SFT checkpoint scored 53.60% on the canonical 500-example 2x2 test set, so it should be treated as a suite adaptation run with poor canonical retention.

## Adversarial Prompt Data

Generate the adversarial prompt supplement:

```bash
.venv/bin/python generate_adversarial_training.py
```

This creates 500 supplemental examples and a 5500-row combined chat file:

```text
data/adversarial/prompt_adv500_seed161804.jsonl
data/adversarial/prompt_adv500_seed161804_chat.jsonl
data/adversarial/train_5500_prompt_adv500_chat.jsonl
```

The supplement avoids matrices already present in `data/train.jsonl`, emphasizes `compact_pairs` and `json_payoffs`, and balances every prompt variant across 0, 1, 2, 3, and 4 equilibria.

Default supplement mix:

| Prompt variant | Examples | Examples per equilibrium-count bucket |
| --- | ---: | ---: |
| `compact_pairs` | 150 | 30 |
| `json_payoffs` | 150 | 30 |
| `standard_table` | 100 | 20 |
| `minimal_matrix` | 50 | 10 |
| `answer_only` | 50 | 10 |

The paper draft records a matrix-overlap audit across public splits. The canonical train/validation/test files are jointly de-duplicated by construction. The adversarial supplement excludes canonical training matrices; one supplement matrix overlaps the balanced stress set, so the adversarial robustness set is the primary adversarial follow-up evidence.

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
  --train-chat data/adversarial/train_5500_prompt_adv500_chat.jsonl \
  --run-name qwen36_27b_sft_5000_plus_prompt_adv500 \
  --out-manifest runs/qwen36_27b_sft_5000_plus_prompt_adv500.json
```

## Repeated-Seed Sweep

Generate the two additional training-data seeds used for the repeated-seed learning curve:

```bash
.venv/bin/python make_repeated_seed_splits.py --seed 1009 --seed 2027
```

This writes seed-specific train/validation data and sweep files under `data/repeated_seeds/`.

Run each seed-specific SFT job using the same Tinker settings. For example:

```bash
.venv/bin/python run_tinker_sft.py \
  --config bench_config.json \
  --train-chat data/repeated_seeds/seed_1009/sweeps/train_1000_chat.jsonl \
  --run-name qwen36_27b_seed1009_sft_1000 \
  --out-manifest runs/qwen36_27b_seed1009_sft_1000.json
```

Evaluate every seed-specific checkpoint on the fixed canonical test set:

```bash
.venv/bin/python run_tinker_predict.py \
  --config bench_config.json \
  --gold data/test.jsonl \
  --model-path "tinker://..." \
  --max-tokens 256 \
  --concurrency 16 \
  --out predictions/qwen36_27b_seed1009_sft_1000.jsonl
```

Score each prediction file with `score_predictions.py`, then summarize the repeated-seed statistics:

```bash
.venv/bin/python summarize_seed_sweep.py \
  --out-json reports/seed_sweep_results.json \
  --out-md reports/seed_sweep_results.md
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
  --adversarial reports/adversarial_results.json \
  --seed-sweep reports/seed_sweep_results.json \
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
  --seed-sweep reports/seed_sweep_results.json \
  --out-dir reports/figures
```

## Paper Build

The arXiv-ready paper source is `paper/gt_bench_paper.tex`. It uses PNG figures under `paper/figures/` and an inline bibliography, so no BibTeX step is required.

Compile with Tectonic from the repository root:

```bash
mkdir -p paper/build
/path/to/tectonic --outdir paper/build paper/gt_bench_paper.tex
```

Create a minimal source archive from the paper directory:

```bash
cd paper
rm -f build/gt_bench_arxiv_source.zip
zip -j build/gt_bench_arxiv_source.zip gt_bench_paper.tex
zip -r build/gt_bench_arxiv_source.zip figures
```

`paper/build/` is ignored by git. Commit the paper source and `paper/figures/*.png`, not generated PDFs or zip files.
