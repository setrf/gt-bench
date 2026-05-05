from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

from generate_dataset import Profile


PROFILE_PATTERN = re.compile(r"\(?\b([UD])\s*,\s*([LR])\b\)?", re.IGNORECASE)
NONE_PATTERN = re.compile(
    r"\b(?:none|no\s+pure(?:-strategy)?\s+nash\s+equilibri(?:um|a))\b",
    re.IGNORECASE,
)
ANSWER_MARKER_PATTERN = re.compile(
    r"(?:final answer\s*:|conclusion\s*:?)",
    re.IGNORECASE,
)
ANSWER_SENTENCE_PATTERN = re.compile(
    r"[^.\n]*(?:pure(?:-strategy)?\s+nash\s+equilibri(?:um|a)|"
    r"nash\s+equilibri(?:um|a))[^.\n]*(?:\.|$)",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def candidate_answer_spans(text: str) -> list[str]:
    spans: list[str] = []
    markers = list(ANSWER_MARKER_PATTERN.finditer(text))
    if markers:
        spans.append(text[markers[-1].end() :])

    sentences = ANSWER_SENTENCE_PATTERN.findall(text)
    spans.extend(reversed(sentences))
    spans.append(text)
    return spans


def parse_profiles(text: str) -> set[Profile]:
    return {
        (row.upper(), col.upper())
        for row, col in PROFILE_PATTERN.findall(text)
    }


def parse_prediction(text: str) -> set[Profile] | None:
    for span in candidate_answer_spans(text):
        none_match = NONE_PATTERN.search(span)
        profile_match = PROFILE_PATTERN.search(span)
        if none_match and (profile_match is None or none_match.start() < profile_match.start()):
            return set()

        profiles = parse_profiles(span)
        if profiles:
            return profiles
        if none_match:
            return set()

    return None


def gold_profiles(row: dict[str, object]) -> set[Profile]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"gold row {row.get('id')} is missing metadata")

    equilibria = metadata.get("pure_nash_equilibria")
    if not isinstance(equilibria, list):
        raise ValueError(f"gold row {row.get('id')} is missing pure_nash_equilibria")

    return {(str(profile[0]), str(profile[1])) for profile in equilibria}


def score(gold_rows: list[dict[str, object]], pred_rows: list[dict[str, object]]) -> dict[str, object]:
    predictions = {
        str(row.get("id")): str(row.get("prediction", ""))
        for row in pred_rows
    }

    correct = 0
    failed_examples: list[dict[str, object]] = []
    by_count: dict[str, dict[str, float | int]] = {}

    for gold_row in gold_rows:
        example_id = str(gold_row["id"])
        gold = gold_profiles(gold_row)
        prediction = predictions.get(example_id, "")
        parsed = parse_prediction(prediction)
        is_correct = parsed is not None and parsed == gold

        bucket = str(len(gold))
        stats = by_count.setdefault(bucket, {"total": 0, "correct": 0, "accuracy": 0.0})
        stats["total"] = int(stats["total"]) + 1

        if is_correct:
            correct += 1
            stats["correct"] = int(stats["correct"]) + 1
        else:
            failed_examples.append(
                {
                    "id": example_id,
                    "prompt": gold_row.get("prompt", ""),
                    "gold": sorted(gold),
                    "prediction": prediction,
                }
            )

    for stats in by_count.values():
        total = int(stats["total"])
        stats["accuracy"] = int(stats["correct"]) / total if total else 0.0

    total = len(gold_rows)
    return {
        "total_examples": total,
        "exact_match_accuracy": correct / total if total else 0.0,
        "num_correct": correct,
        "num_incorrect": total - correct,
        "accuracy_by_number_of_equilibria": by_count,
        "failed_examples": failed_examples,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score GT-Bench model predictions.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = score(read_jsonl(args.gold), read_jsonl(args.pred))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
