from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from score_predictions import gold_profiles, parse_prediction, read_jsonl


def empty_stats() -> dict[str, float | int]:
    return {"total": 0, "correct": 0, "accuracy": 0.0}


def update_stats(stats: dict[str, float | int], is_correct: bool) -> None:
    stats["total"] = int(stats["total"]) + 1
    if is_correct:
        stats["correct"] = int(stats["correct"]) + 1


def finalize_stats(groups: dict[str, dict[str, float | int]]) -> None:
    for stats in groups.values():
        total = int(stats["total"])
        stats["accuracy"] = int(stats["correct"]) / total if total else 0.0


def metadata_value(row: dict[str, object], key: str) -> object:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"gold row {row.get('id')} is missing metadata")
    if key not in metadata:
        raise ValueError(f"gold row {row.get('id')} is missing metadata.{key}")
    return metadata[key]


def score_robustness(
    gold_rows: list[dict[str, object]],
    pred_rows: list[dict[str, object]],
) -> dict[str, object]:
    predictions = {
        str(row.get("id")): str(row.get("prediction", ""))
        for row in pred_rows
    }

    correct = 0
    by_variant: dict[str, dict[str, float | int]] = {}
    by_count: dict[str, dict[str, float | int]] = {}
    by_variant_and_count: dict[str, dict[str, dict[str, float | int]]] = {}
    failed_examples: list[dict[str, object]] = []

    for gold_row in gold_rows:
        example_id = str(gold_row["id"])
        variant = str(metadata_value(gold_row, "prompt_variant"))
        gold = gold_profiles(gold_row)
        count = str(len(gold))
        prediction = predictions.get(example_id, "")
        parsed = parse_prediction(prediction)
        is_correct = parsed is not None and parsed == gold

        update_stats(by_variant.setdefault(variant, empty_stats()), is_correct)
        update_stats(by_count.setdefault(count, empty_stats()), is_correct)
        variant_counts = by_variant_and_count.setdefault(variant, {})
        update_stats(variant_counts.setdefault(count, empty_stats()), is_correct)

        if is_correct:
            correct += 1
        else:
            failed_examples.append(
                {
                    "id": example_id,
                    "prompt_variant": variant,
                    "equilibrium_count": int(count),
                    "prompt": gold_row.get("prompt", ""),
                    "gold": sorted(gold),
                    "prediction": prediction,
                }
            )

    finalize_stats(by_variant)
    finalize_stats(by_count)
    for variant_counts in by_variant_and_count.values():
        finalize_stats(variant_counts)

    total = len(gold_rows)
    return {
        "total_examples": total,
        "exact_match_accuracy": correct / total if total else 0.0,
        "num_correct": correct,
        "num_incorrect": total - correct,
        "accuracy_by_prompt_variant": by_variant,
        "accuracy_by_number_of_equilibria": by_count,
        "accuracy_by_variant_and_number_of_equilibria": by_variant_and_count,
        "failed_examples": failed_examples,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score GT-Bench prompt-robustness predictions.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = score_robustness(read_jsonl(args.gold), read_jsonl(args.pred))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
