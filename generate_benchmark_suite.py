from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from game_theory_suite import (
    DEFAULT_SUITE_FAMILIES,
    generate_suite_examples,
    split_suite_examples,
    to_suite_chat_row,
)
from generate_dataset import write_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the broader GT-Bench game-theory task suite."
    )
    parser.add_argument("--per-family", type=int, default=50)
    parser.add_argument("--train-per-family", type=int, default=None)
    parser.add_argument("--val-per-family", type=int, default=None)
    parser.add_argument("--test-per-family", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_SUITE_FAMILIES))
    parser.add_argument("--max-attempts", type=int, default=5_000_000)
    parser.add_argument("--out", type=Path, default=Path("data/suite/gt_bench_suite.jsonl"))
    parser.add_argument(
        "--chat-out",
        type=Path,
        default=Path("data/suite/gt_bench_suite_chat.jsonl"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/suite"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.train_per_family is not None
        or args.val_per_family is not None
        or args.test_per_family is not None
    ):
        train = int(args.train_per_family or 0)
        val = int(args.val_per_family or 0)
        test = int(args.test_per_family or 0)
        splits = split_suite_examples(
            train_per_family=train,
            val_per_family=val,
            test_per_family=test,
            seed=args.seed,
            families=args.families,
            max_attempts=args.max_attempts,
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for split_name, rows in splits.items():
            write_jsonl(args.out_dir / f"{split_name}.jsonl", rows)
            write_jsonl(
                args.out_dir / f"{split_name}_chat.jsonl",
                (to_suite_chat_row(example) for example in rows),
            )
            print(f"wrote {len(rows)} {split_name} examples to {args.out_dir}")
        return 0

    examples = generate_suite_examples(
        per_family=args.per_family,
        seed=args.seed,
        families=args.families,
        max_attempts=args.max_attempts,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.chat_out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, examples)
    write_jsonl(args.chat_out, (to_suite_chat_row(example) for example in examples))
    print(f"wrote {len(examples)} suite examples to {args.out}")
    print(f"wrote {len(examples)} suite chat examples to {args.chat_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
