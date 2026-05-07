# GT-Bench Adversarial SFT Follow-Up

## Summary

This follow-up targets the only major weakness left after the `v0.1.3` robustness evaluation: prompt format sensitivity.

The original 5000-example SFT checkpoint improved prompt robustness from 64.00% to 88.40%, but remained weaker on:

- `compact_pairs`: 78.00%
- `json_payoffs`: 68.00%

The adversarial follow-up keeps the task exactly the same: identify all pure-strategy Nash equilibria in a 2x2 normal-form game. It only changes the training prompt surfaces.

## Supplemental Data

`generate_adversarial_training.py` creates a deterministic 1000-example supplement:

| Prompt variant | Examples | Per equilibrium-count bucket |
| --- | ---: | ---: |
| `compact_pairs` | 300 | 60 |
| `json_payoffs` | 300 | 60 |
| `minimal_matrix` | 150 | 30 |
| `answer_only` | 150 | 30 |
| `standard_table` | 100 | 20 |

The generator avoids duplicate matrices from `data/train.jsonl`, balances each variant across 0, 1, 2, 3, and 4 equilibria, and writes ignored local data files under `data/adversarial/`.

```bash
.venv/bin/python generate_adversarial_training.py
```

The combined SFT file is:

```text
data/adversarial/train_6000_prompt_adv_chat.jsonl
```

## Training

Run one additional LoRA SFT job:

```bash
.venv/bin/python run_tinker_sft.py \
  --config bench_config.json \
  --train-chat data/adversarial/train_6000_prompt_adv_chat.jsonl \
  --run-name qwen36_27b_sft_5000_plus_prompt_adv \
  --out-manifest runs/qwen36_27b_sft_5000_plus_prompt_adv.json
```

Do not commit the manifest. It contains private checkpoint paths.

## Evaluation

Evaluate the adversarial checkpoint on the same public evaluation sets:

- canonical test
- independent confirmation set
- balanced stress set
- prompt-robustness set

The public result tracker is generated with:

```bash
.venv/bin/python summarize_adversarial.py
.venv/bin/python plot_results.py \
  --summary reports/gt_bench_results.json \
  --robustness reports/robustness_results.json \
  --adversarial reports/adversarial_results.json \
  --out-dir reports/figures
```

Current status is recorded in:

- `reports/adversarial_results.md`
- `reports/adversarial_results.json`
- `reports/figures/adversarial_comparison.svg`

## Acceptance Criteria

- Robustness accuracy reaches at least 95.00%.
- Canonical accuracy does not regress by more than 0.50 percentage points versus the original 5000-example SFT checkpoint.
- `compact_pairs` and `json_payoffs` improve clearly.

## Claim Boundaries

This follow-up can support a targeted robustness claim for one synthetic formal task. It still does not claim broad game-theory reasoning and does not test mixed strategies, dominance, welfare, sequential games, or story problems.
