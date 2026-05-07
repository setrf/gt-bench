from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from generate_dataset import (
    PROFILES,
    PayoffMatrix,
    build_example,
    find_pure_nash_equilibria,
    payoffs_from_values,
    to_chat_row,
    write_jsonl,
)
from generate_robustness_set import PROMPT_VARIANTS, RENDERERS
from tinker_common import jsonl_rows


DEFAULT_VARIANT_COUNTS = {
    "compact_pairs": 150,
    "json_payoffs": 150,
    "standard_table": 100,
    "minimal_matrix": 50,
    "answer_only": 50,
}
DEFAULT_COUNTS = (0, 1, 2, 3, 4)


def payoffs_signature(payoffs: PayoffMatrix) -> tuple[int, ...]:
    return tuple(value for profile in PROFILES for value in payoffs[profile])


def metadata_signature(row: dict[str, object]) -> tuple[int, ...]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"row {row.get('id')} is missing metadata")
    payoffs = metadata.get("payoffs")
    if not isinstance(payoffs, dict):
        raise ValueError(f"row {row.get('id')} is missing metadata.payoffs")
    values: list[int] = []
    for row_name, col_name in PROFILES:
        cell = payoffs.get(f"{row_name},{col_name}")
        if not isinstance(cell, list | tuple) or len(cell) != 2:
            raise ValueError(f"row {row.get('id')} has malformed payoff cell {row_name},{col_name}")
        values.extend([int(cell[0]), int(cell[1])])
    return tuple(values)


def load_excluded_signatures(paths: Iterable[Path]) -> set[tuple[int, ...]]:
    signatures: set[tuple[int, ...]] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in jsonl_rows(path):
            signatures.add(metadata_signature(row))
    return signatures


def parse_variant_count(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--variant-count values must look like variant=count")
    variant, count_text = value.split("=", 1)
    if variant not in PROMPT_VARIANTS:
        raise argparse.ArgumentTypeError(f"unknown prompt variant: {variant}")
    try:
        count = int(count_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid count for {variant}: {count_text}") from exc
    if count <= 0:
        raise argparse.ArgumentTypeError("variant counts must be positive")
    return variant, count


def bucket_targets(
    variant_counts: dict[str, int],
    counts: Sequence[int],
) -> dict[tuple[str, int], int]:
    if not counts:
        raise ValueError("counts cannot be empty")
    targets: dict[tuple[str, int], int] = {}
    for variant, total in variant_counts.items():
        if variant not in PROMPT_VARIANTS:
            raise ValueError(f"unknown prompt variant: {variant}")
        if total % len(counts) != 0:
            raise ValueError(f"{variant} total {total} is not divisible by {len(counts)} buckets")
        per_count = total // len(counts)
        for count in counts:
            targets[(variant, count)] = per_count
    return targets


def generate_adversarial_examples(
    seed: int,
    variant_counts: dict[str, int] | None = None,
    counts: Sequence[int] = DEFAULT_COUNTS,
    exclude_signatures: set[tuple[int, ...]] | None = None,
    max_attempts: int = 5_000_000,
) -> list[dict[str, object]]:
    variant_counts = dict(variant_counts or DEFAULT_VARIANT_COUNTS)
    targets = bucket_targets(variant_counts, counts)
    rng = random.Random(seed)
    seen = set(exclude_signatures or set())
    needed: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    ordered_variants = list(variant_counts)
    wanted_counts = set(counts)
    attempts = 0

    def complete() -> bool:
        return all(len(needed[key]) >= target for key, target in targets.items())

    while not complete():
        attempts += 1
        if attempts > max_attempts:
            missing = {
                f"{variant}:{count}": targets[(variant, count)] - len(needed[(variant, count)])
                for variant in ordered_variants
                for count in counts
                if len(needed[(variant, count)]) < targets[(variant, count)]
            }
            raise RuntimeError(f"hit max_attempts={max_attempts}; missing {missing}")

        values = tuple(rng.randint(0, 9) for _ in range(8))
        if values in seen:
            continue
        payoffs = payoffs_from_values(values)
        count = len(find_pure_nash_equilibria(payoffs))
        if count not in wanted_counts:
            continue

        open_variants = [
            variant
            for variant in ordered_variants
            if len(needed[(variant, count)]) < targets[(variant, count)]
        ]
        if not open_variants:
            seen.add(values)
            continue

        variant = open_variants[0]
        seen.add(values)
        index = len(needed[(variant, count)]) + 1
        example_id = f"adv_{variant}_eq{count}_{index:04d}"
        example = build_example(payoffs, example_id)
        example["prompt"] = RENDERERS[variant](payoffs)
        metadata = dict(example["metadata"])  # type: ignore[arg-type]
        metadata["prompt_variant"] = variant
        metadata["equilibrium_count"] = count
        metadata["adversarial_training"] = True
        example["metadata"] = metadata
        needed[(variant, count)].append(example)

    examples: list[dict[str, object]] = []
    for variant in ordered_variants:
        for count in counts:
            examples.extend(needed[(variant, count)])
    return examples


def to_adversarial_chat_row(example: dict[str, object]) -> dict[str, object]:
    metadata = example.get("metadata")
    variant = metadata.get("prompt_variant") if isinstance(metadata, dict) else None
    if variant == "answer_only":
        return {
            "messages": [
                {"role": "user", "content": str(example["prompt"])},
                {"role": "assistant", "content": f"Final answer: {example['answer']}"},
            ]
        }
    return to_chat_row(example)


def combine_chat_files(base_chat: Path, supplement_chat: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_path.open("w", encoding="utf-8") as out_handle:
        for source in (base_chat, supplement_chat):
            with source.open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    if not line.strip():
                        continue
                    out_handle.write(line)
                    total += 1
    return total


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GT-Bench adversarial prompt SFT data.")
    parser.add_argument("--seed", type=int, default=161804)
    parser.add_argument("--base", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--base-chat", type=Path, default=Path("data/train_chat.jsonl"))
    parser.add_argument(
        "--variant-count",
        action="append",
        type=parse_variant_count,
        default=[],
        help="Override a default variant count, e.g. compact_pairs=150.",
    )
    parser.add_argument("--counts", type=int, nargs="+", default=list(DEFAULT_COUNTS))
    parser.add_argument("--max-attempts", type=int, default=5_000_000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/adversarial/prompt_adv500_seed161804.jsonl"),
    )
    parser.add_argument(
        "--chat-out",
        type=Path,
        default=Path("data/adversarial/prompt_adv500_seed161804_chat.jsonl"),
    )
    parser.add_argument(
        "--combined-chat-out",
        type=Path,
        default=Path("data/adversarial/train_5500_prompt_adv500_chat.jsonl"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    variant_counts = dict(DEFAULT_VARIANT_COUNTS)
    variant_counts.update(dict(args.variant_count))
    excluded = load_excluded_signatures([args.base])
    examples = generate_adversarial_examples(
        seed=args.seed,
        variant_counts=variant_counts,
        counts=args.counts,
        exclude_signatures=excluded,
        max_attempts=args.max_attempts,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.chat_out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, examples)
    write_jsonl(args.chat_out, (to_adversarial_chat_row(example) for example in examples))
    combined_count = combine_chat_files(args.base_chat, args.chat_out, args.combined_chat_out)
    print(f"wrote {len(examples)} adversarial examples to {args.out}")
    print(f"wrote {len(examples)} adversarial chat examples to {args.chat_out}")
    print(f"wrote {combined_count} combined chat examples to {args.combined_chat_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
