from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Sequence


ROWS = ("U", "D")
COLS = ("L", "R")
PROFILES = (("U", "L"), ("U", "R"), ("D", "L"), ("D", "R"))
MAX_UNIQUE_MATRICES = 10**8

PayoffCell = tuple[int, int]
Profile = tuple[str, str]
PayoffMatrix = dict[Profile, PayoffCell]


def make_payoffs(
    ul: PayoffCell,
    ur: PayoffCell,
    dl: PayoffCell,
    dr: PayoffCell,
) -> PayoffMatrix:
    return {
        ("U", "L"): ul,
        ("U", "R"): ur,
        ("D", "L"): dl,
        ("D", "R"): dr,
    }


def payoffs_from_values(values: Sequence[int]) -> PayoffMatrix:
    if len(values) != 8:
        raise ValueError("a 2x2 two-player matrix needs exactly 8 payoff values")
    cells = [
        (values[0], values[1]),
        (values[2], values[3]),
        (values[4], values[5]),
        (values[6], values[7]),
    ]
    return dict(zip(PROFILES, cells))


def best_responses(payoffs: PayoffMatrix) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    p1_best: dict[str, tuple[str, ...]] = {}
    p2_best: dict[str, tuple[str, ...]] = {}

    for col in COLS:
        values = {row: payoffs[(row, col)][0] for row in ROWS}
        best_value = max(values.values())
        p1_best[col] = tuple(row for row in ROWS if values[row] == best_value)

    for row in ROWS:
        values = {col: payoffs[(row, col)][1] for col in COLS}
        best_value = max(values.values())
        p2_best[row] = tuple(col for col in COLS if values[col] == best_value)

    return p1_best, p2_best


def find_pure_nash_equilibria(payoffs: PayoffMatrix) -> list[Profile]:
    p1_best, p2_best = best_responses(payoffs)
    return [
        (row, col)
        for row, col in PROFILES
        if row in p1_best[col] and col in p2_best[row]
    ]


def format_prompt(payoffs: PayoffMatrix) -> str:
    def cell(row: str, col: str) -> str:
        p1, p2 = payoffs[(row, col)]
        return f"({p1},{p2})"

    return (
        "Player 1 chooses U or D. Player 2 chooses L or R. Payoffs are shown as "
        "(Player 1 payoff, Player 2 payoff):\n\n"
        "        L       R\n"
        f"U    {cell('U', 'L'):<7} {cell('U', 'R')}\n"
        f"D    {cell('D', 'L'):<7} {cell('D', 'R')}\n\n"
        "Find all pure-strategy Nash equilibria. Explain briefly."
    )


def format_profile(profile: Profile) -> str:
    return f"({profile[0]}, {profile[1]})"


def join_phrases(items: Sequence[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def format_answer(equilibria: Sequence[Profile]) -> str:
    profiles = [format_profile(profile) for profile in equilibria]
    if not profiles:
        return "There are no pure-strategy Nash equilibria."
    if len(profiles) == 1:
        return f"The pure-strategy Nash equilibrium is {profiles[0]}."
    return f"The pure-strategy Nash equilibria are {join_phrases(profiles)}."


def _best_response_clause(player: str, target: str, responses: Sequence[str]) -> str:
    noun = "best response" if len(responses) == 1 else "best responses"
    verb = "is" if len(responses) == 1 else "are"
    return f"{player}'s {noun} to {target} {verb} {join_phrases(list(responses))}"


def format_solution(payoffs: PayoffMatrix) -> str:
    p1_best, p2_best = best_responses(payoffs)
    equilibria = find_pure_nash_equilibria(payoffs)

    p1_text = (
        f"{_best_response_clause('Player 1', 'L', p1_best['L'])}, and "
        f"{_best_response_clause('Player 1', 'R', p1_best['R'])}."
    )
    p2_text = (
        f"{_best_response_clause('Player 2', 'U', p2_best['U'])}, and "
        f"{_best_response_clause('Player 2', 'D', p2_best['D'])}."
    )

    profiles = [format_profile(profile) for profile in equilibria]
    if not profiles:
        conclusion = "Therefore there are no mutual best-response profiles."
    elif len(profiles) == 1:
        conclusion = f"Therefore the mutual best-response profile is {profiles[0]}."
    else:
        conclusion = f"Therefore the mutual best-response profiles are {join_phrases(profiles)}."

    return f"{p1_text} {p2_text} {conclusion}"


def metadata_for(payoffs: PayoffMatrix, equilibria: Sequence[Profile]) -> dict[str, object]:
    return {
        "payoffs": {
            f"{row},{col}": [payoffs[(row, col)][0], payoffs[(row, col)][1]]
            for row, col in PROFILES
        },
        "pure_nash_equilibria": [[row, col] for row, col in equilibria],
    }


def build_example(payoffs: PayoffMatrix, example_id: str) -> dict[str, object]:
    equilibria = find_pure_nash_equilibria(payoffs)
    return {
        "id": example_id,
        "prompt": format_prompt(payoffs),
        "answer": format_answer(equilibria),
        "solution": format_solution(payoffs),
        "metadata": metadata_for(payoffs, equilibria),
    }


def to_chat_row(example: dict[str, object]) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": str(example["prompt"])},
            {
                "role": "assistant",
                "content": (
                    f"Final answer: {example['answer']}\n\n"
                    f"Reasoning: {example['solution']}"
                ),
            },
        ]
    }


def generate_examples(total: int, seed: int) -> list[dict[str, object]]:
    if total > MAX_UNIQUE_MATRICES:
        raise ValueError(f"cannot generate more than {MAX_UNIQUE_MATRICES} unique matrices")

    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    examples: list[dict[str, object]] = []

    while len(examples) < total:
        values = tuple(rng.randint(0, 9) for _ in range(8))
        if values in seen:
            continue
        seen.add(values)
        payoffs = payoffs_from_values(values)
        example_id = f"example_{len(examples) + 1:06d}"
        examples.append(build_example(payoffs, example_id))

    return examples


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_splits(out_dir: Path, train: int, val: int, test: int, seed: int) -> None:
    if min(train, val, test) < 0:
        raise ValueError("split sizes must be nonnegative")

    out_dir.mkdir(parents=True, exist_ok=True)
    examples = generate_examples(train + val + test, seed)
    splits = {
        "train": examples[:train],
        "val": examples[train : train + val],
        "test": examples[train + val :],
    }

    for split_name, rows in splits.items():
        write_jsonl(out_dir / f"{split_name}.jsonl", rows)
        write_jsonl(out_dir / f"{split_name}_chat.jsonl", (to_chat_row(row) for row in rows))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GT-Bench 2x2 Nash examples.")
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--val", type=int, default=500)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    write_splits(args.out, args.train, args.val, args.test, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

