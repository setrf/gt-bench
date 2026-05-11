from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from game_theory_suite import generate_suite_examples, to_suite_chat_row
from generate_dataset import write_jsonl


DEFAULT_FAMILIES = (
    "mixed_2x2",
    "dominance",
    "large_normal_form",
    "extensive_form",
    "natural_language",
    "repeated_interaction",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the broader GT-Bench game-theory task suite."
    )
    parser.add_argument("--per-family", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--max-attempts", type=int, default=5_000_000)
    parser.add_argument("--out", type=Path, default=Path("data/suite/gt_bench_suite.jsonl"))
    parser.add_argument(
        "--chat-out",
        type=Path,
        default=Path("data/suite/gt_bench_suite_chat.jsonl"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
