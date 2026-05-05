# GT-Bench: A Minimal Game-Theory Fine-Tuning Benchmark

## Project overview

GT-Bench is a minimal Tinker fine-tuning benchmark for strategic reasoning. It generates small, fully verifiable 2x2 normal-form games and asks a model to find all pure-strategy Nash equilibria with a brief explanation.

The repository is intentionally compact: one task, one exact solver, one dataset generator, one scorer, and focused tests.

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

## How to fine-tune

Use `data/train_chat.jsonl` as the fine-tuning file on Tinker. Each row contains a user message with the matrix prompt and an assistant message with:

- a final answer listing the pure-strategy Nash equilibria
- concise reasoning based on exact best responses

Use `data/val_chat.jsonl` as a held-out validation file if the fine-tuning workflow supports it.

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

## Expected result

The expected demonstration is an increase in exact-match accuracy on held-out 2x2 Nash equilibrium problems after fine-tuning.

## Limitations

GT-Bench is deliberately narrow. It uses synthetic data, covers pure equilibria only, and does not prove broad game-theory reasoning improvement.

It is best understood as a controlled fine-tuning benchmark for one formal reasoning task, not as a general game-theory benchmark.

## Testing

Run:

```bash
.venv/bin/python -m pytest -q
```

The tests cover a Prisoner's Dilemma style one-equilibrium game, a coordination game with two equilibria, a matching pennies style game with no pure equilibrium, a tie case with multiple best responses, and common prediction parser formats.
