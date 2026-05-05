# Release Notes

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
