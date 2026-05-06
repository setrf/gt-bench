from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

from generate_dataset import (
    COLS,
    ROWS,
    PayoffMatrix,
    build_example,
    find_pure_nash_equilibria,
    format_prompt,
    payoffs_from_values,
    to_chat_row,
    write_jsonl,
)


PROMPT_VARIANTS = (
    "standard_table",
    "compact_pairs",
    "json_payoffs",
    "minimal_matrix",
    "answer_only",
)


def compact_pairs_prompt(payoffs: PayoffMatrix) -> str:
    parts = [
        f"{row},{col}=({payoffs[(row, col)][0]},{payoffs[(row, col)][1]})"
        for row in ROWS
        for col in COLS
    ]
    return (
        "Player 1 chooses U or D; Player 2 chooses L or R. "
        "Payoffs are (Player 1 payoff, Player 2 payoff): "
        f"{'; '.join(parts)}. "
        "Find all pure-strategy Nash equilibria. Explain briefly."
    )


def json_payoffs_prompt(payoffs: PayoffMatrix) -> str:
    payload = {
        f"{row},{col}": [payoffs[(row, col)][0], payoffs[(row, col)][1]]
        for row in ROWS
        for col in COLS
    }
    return (
        "For this 2x2 normal-form game, Player 1 has strategies U,D and Player 2 has "
        "strategies L,R. Payoffs are [Player 1, Player 2]:\n\n"
        f"{json.dumps(payload, sort_keys=True)}\n\n"
        "Find all pure-strategy Nash equilibria. Explain briefly."
    )


def minimal_matrix_prompt(payoffs: PayoffMatrix) -> str:
    def cell(row: str, col: str) -> str:
        return f"{payoffs[(row, col)][0]},{payoffs[(row, col)][1]}"

    return (
        "2x2 game. Rows U,D. Columns L,R. Entries are P1,P2.\n"
        f"      L     R\n"
        f"U   {cell('U', 'L'):<5} {cell('U', 'R')}\n"
        f"D   {cell('D', 'L'):<5} {cell('D', 'R')}\n"
        "List all pure Nash equilibria."
    )


def answer_only_prompt(payoffs: PayoffMatrix) -> str:
    return (
        f"{format_prompt(payoffs)}\n\n"
        "Return only the final answer. Do not include reasoning."
    )


RENDERERS: dict[str, Callable[[PayoffMatrix], str]] = {
    "standard_table": format_prompt,
    "compact_pairs": compact_pairs_prompt,
    "json_payoffs": json_payoffs_prompt,
    "minimal_matrix": minimal_matrix_prompt,
    "answer_only": answer_only_prompt,
}


def generate_robustness_examples(
    per_bucket: int,
    seed: int,
    variants: Sequence[str],
    counts: Sequence[int],
    max_attempts: int,
) -> list[dict[str, object]]:
    if per_bucket <= 0:
        raise ValueError("per_bucket must be positive")
    if not variants:
        raise ValueError("at least one prompt variant is required")
    if not counts:
        raise ValueError("at least one equilibrium count is required")
    unknown = sorted(set(variants) - set(RENDERERS))
    if unknown:
        raise ValueError(f"unknown prompt variants: {', '.join(unknown)}")

    rng = random.Random(seed)
    wanted_counts = set(counts)
    needed: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[int, ...]] = set()
    attempts = 0

    def complete() -> bool:
        return all(
            len(needed[(variant, count)]) >= per_bucket
            for variant in variants
            for count in wanted_counts
        )

    while not complete():
        attempts += 1
        if attempts > max_attempts:
            missing = {
                f"{variant}:{count}": per_bucket - len(needed[(variant, count)])
                for variant in variants
                for count in sorted(wanted_counts)
                if len(needed[(variant, count)]) < per_bucket
            }
            raise RuntimeError(f"hit max_attempts={max_attempts}; missing {missing}")

        values = tuple(rng.randint(0, 9) for _ in range(8))
        if values in seen:
            continue
        seen.add(values)

        payoffs = payoffs_from_values(values)
        count = len(find_pure_nash_equilibria(payoffs))
        if count not in wanted_counts:
            continue

        open_variants = [
            variant
            for variant in variants
            if len(needed[(variant, count)]) < per_bucket
        ]
        if not open_variants:
            continue

        variant = open_variants[0]
        index = len(needed[(variant, count)]) + 1
        example_id = f"robust_{variant}_eq{count}_{index:04d}"
        example = build_example(payoffs, example_id)
        example["prompt"] = RENDERERS[variant](payoffs)
        metadata = dict(example["metadata"])  # type: ignore[arg-type]
        metadata["prompt_variant"] = variant
        metadata["equilibrium_count"] = count
        example["metadata"] = metadata
        needed[(variant, count)].append(example)

    examples: list[dict[str, object]] = []
    for variant in variants:
        for count in sorted(wanted_counts):
            examples.extend(needed[(variant, count)])
    return examples


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GT-Bench prompt-robustness sets.")
    parser.add_argument("--per-bucket", type=int, default=10)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--variants", nargs="+", default=list(PROMPT_VARIANTS))
    parser.add_argument("--counts", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-attempts", type=int, default=5_000_000)
    parser.add_argument("--out", type=Path, default=Path("data/robust/robust_seed271828.jsonl"))
    parser.add_argument(
        "--chat-out",
        type=Path,
        default=Path("data/robust/robust_seed271828_chat.jsonl"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    examples = generate_robustness_examples(
        per_bucket=args.per_bucket,
        seed=args.seed,
        variants=args.variants,
        counts=args.counts,
        max_attempts=args.max_attempts,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.chat_out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, examples)
    write_jsonl(args.chat_out, (to_chat_row(example) for example in examples))
    print(f"wrote {len(examples)} examples to {args.out}")
    print(f"wrote {len(examples)} chat examples to {args.chat_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
