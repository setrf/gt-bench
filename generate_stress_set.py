from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from generate_dataset import (
    build_example,
    find_pure_nash_equilibria,
    payoffs_from_values,
    to_chat_row,
    write_jsonl,
)


def generate_balanced_examples(
    per_count: int,
    seed: int,
    counts: Sequence[int],
    max_attempts: int,
) -> list[dict[str, object]]:
    if per_count <= 0:
        raise ValueError("per_count must be positive")
    if not counts:
        raise ValueError("at least one equilibrium count is required")

    rng = random.Random(seed)
    wanted = set(counts)
    buckets: dict[int, list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[int, ...]] = set()
    attempts = 0

    while any(len(buckets[count]) < per_count for count in wanted):
        attempts += 1
        if attempts > max_attempts:
            missing = {
                count: per_count - len(buckets[count])
                for count in sorted(wanted)
                if len(buckets[count]) < per_count
            }
            raise RuntimeError(f"hit max_attempts={max_attempts}; missing {missing}")

        values = tuple(rng.randint(0, 9) for _ in range(8))
        if values in seen:
            continue
        seen.add(values)

        payoffs = payoffs_from_values(values)
        count = len(find_pure_nash_equilibria(payoffs))
        if count not in wanted or len(buckets[count]) >= per_count:
            continue

        example_id = f"stress_eq{count}_{len(buckets[count]) + 1:04d}"
        buckets[count].append(build_example(payoffs, example_id))

    examples: list[dict[str, object]] = []
    for count in sorted(wanted):
        examples.extend(buckets[count])
    return examples


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate balanced GT-Bench stress sets.")
    parser.add_argument("--per-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--counts", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-attempts", type=int, default=5_000_000)
    parser.add_argument("--out", type=Path, default=Path("data/stress/tie_stress_seed314159.jsonl"))
    parser.add_argument(
        "--chat-out",
        type=Path,
        default=Path("data/stress/tie_stress_seed314159_chat.jsonl"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    examples = generate_balanced_examples(
        per_count=args.per_count,
        seed=args.seed,
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
