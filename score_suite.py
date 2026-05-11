from __future__ import annotations

import argparse
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

from game_theory_suite import Profile
from score_predictions import candidate_answer_spans, read_jsonl


NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
MISSING_PATTERN = re.compile(r"\b(?:none|no\s+(?:pure\s+)?(?:nash\s+)?equilibria?)\b", re.I)


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


def metadata(row: dict[str, object]) -> dict[str, Any]:
    value = row.get("metadata")
    if not isinstance(value, dict):
        raise ValueError(f"gold row {row.get('id')} is missing metadata")
    return value


def task_family(row: dict[str, object]) -> str:
    family = metadata(row).get("task_family")
    if not isinstance(family, str):
        raise ValueError(f"gold row {row.get('id')} is missing metadata.task_family")
    return family


def parse_number(value: str) -> Fraction | None:
    value = value.strip()
    if "/" in value:
        try:
            return Fraction(value)
        except ZeroDivisionError:
            return None
    try:
        return Fraction(float(value)).limit_denominator(1000)
    except ValueError:
        return None


def labels_regex(labels: Sequence[str]) -> str:
    return "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))


def extract_labels(text: str, labels: Sequence[str]) -> set[str]:
    pattern = re.compile(rf"\b({labels_regex(labels)})\b", re.I)
    by_lower = {label.lower(): label for label in labels}
    return {by_lower[match.group(1).lower()] for match in pattern.finditer(text)}


def parse_profiles(text: str, rows: Sequence[str], cols: Sequence[str]) -> set[Profile] | None:
    row_pattern = labels_regex(rows)
    col_pattern = labels_regex(cols)
    by_row = {row.lower(): row for row in rows}
    by_col = {col.lower(): col for col in cols}
    pair_pattern = re.compile(
        rf"\(?\b({row_pattern})\b\s*,\s*\b({col_pattern})\b\)?",
        re.I,
    )
    for span in candidate_answer_spans(text):
        if MISSING_PATTERN.search(span) and not pair_pattern.search(span):
            return set()
        matches = pair_pattern.findall(span)
        if matches:
            return {(by_row[row.lower()], by_col[col.lower()]) for row, col in matches}
    return None


def gold_profiles(row: dict[str, object]) -> set[Profile]:
    equilibria = metadata(row).get("pure_nash_equilibria")
    if not isinstance(equilibria, list):
        raise ValueError(f"gold row {row.get('id')} is missing pure_nash_equilibria")
    return {(str(profile[0]), str(profile[1])) for profile in equilibria}


def parse_probability(text: str, label: str) -> Fraction | None:
    pattern = re.compile(
        rf"\b{re.escape(label)}\b\s*(?:=|:)\s*({NUMBER_PATTERN.pattern})",
        re.I,
    )
    match = pattern.search(text)
    if not match:
        return None
    return parse_number(match.group(1))


def parse_mixed_prediction(text: str, row: dict[str, object]) -> dict[str, dict[str, Fraction]] | None:
    gold = metadata(row).get("mixed_nash_equilibrium")
    if not isinstance(gold, dict):
        raise ValueError(f"gold row {row.get('id')} is missing mixed_nash_equilibrium")

    parsed: dict[str, dict[str, Fraction]] = {"P1": {}, "P2": {}}
    for player, strategies in gold.items():
        if not isinstance(strategies, dict):
            raise ValueError(f"gold row {row.get('id')} has malformed mixed_nash_equilibrium")
        for strategy in strategies:
            probability = parse_probability(text, str(strategy))
            if probability is not None:
                parsed[str(player)][str(strategy)] = probability

    for player, strategies in gold.items():
        strategy_names = [str(strategy) for strategy in strategies]
        if len(strategy_names) != 2:
            return None
        known = parsed[str(player)]
        if len(known) == 1:
            missing = [strategy for strategy in strategy_names if strategy not in known]
            known[missing[0]] = 1 - next(iter(known.values()))
        if set(known) != set(strategy_names):
            return None
    return parsed


def mixed_matches(
    parsed: dict[str, dict[str, Fraction]] | None,
    row: dict[str, object],
    tolerance: Fraction = Fraction(1, 100),
) -> bool:
    if parsed is None:
        return False
    gold = metadata(row)["mixed_nash_equilibrium"]
    if not isinstance(gold, dict):
        return False
    for player, strategies in gold.items():
        if not isinstance(strategies, dict):
            return False
        for strategy, value in strategies.items():
            target = Fraction(str(value))
            predicted = parsed.get(str(player), {}).get(str(strategy))
            if predicted is None or abs(predicted - target) > tolerance:
                return False
    return True


def parse_remaining(text: str, label: str, allowed: Sequence[str]) -> set[str] | None:
    pattern = re.compile(
        rf"remaining\s+{re.escape(label)}(?:\s+strategies)?\s*(?:=|:)\s*([^.;\n]+)",
        re.I,
    )
    for span in candidate_answer_spans(text):
        match = pattern.search(span)
        if match:
            found = extract_labels(match.group(1), allowed)
            return found
    return None


def dominance_matches(text: str, row: dict[str, object]) -> bool:
    meta = metadata(row)
    elimination = meta.get("iterated_elimination")
    if not isinstance(elimination, dict):
        raise ValueError(f"gold row {row.get('id')} is missing iterated_elimination")
    rows = [str(value) for value in meta.get("rows", [])]
    cols = [str(value) for value in meta.get("cols", [])]
    predicted_rows = parse_remaining(text, "P1", rows)
    predicted_cols = parse_remaining(text, "P2", cols)
    return (
        predicted_rows == {str(value) for value in elimination.get("remaining_rows", [])}
        and predicted_cols == {str(value) for value in elimination.get("remaining_cols", [])}
    )


def extensive_matches(text: str, row: dict[str, object]) -> bool:
    target = metadata(row).get("subgame_perfect_equilibrium")
    if not isinstance(target, dict):
        raise ValueError(f"gold row {row.get('id')} is missing subgame_perfect_equilibrium")
    patterns = {
        "p1": r"\bP1\s*(?:=|:)\s*(Left|Right)\b",
        "p2_left": r"\bP2\s+after\s+Left\s*(?:=|:)\s*(A|B)\b",
        "p2_right": r"\bP2\s+after\s+Right\s*(?:=|:)\s*(A|B)\b",
        "path": r"\bPath\s*(?:=|:)\s*(Left|Right)\s*[-,]\s*(A|B)\b",
    }
    parsed: dict[str, str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.I)
        if not match:
            return False
        parsed[key] = "-".join(match.groups()) if key == "path" else match.group(1)
    return all(parsed[key].lower() == str(target[key]).lower() for key in patterns)


def repeated_matches(text: str, row: dict[str, object]) -> bool:
    target = metadata(row).get("simulation")
    if not isinstance(target, dict):
        raise ValueError(f"gold row {row.get('id')} is missing simulation")
    p1_pattern = re.compile(r"\bP1\s+payoff\s*(?:=|:)\s*(-?\d+)\b", re.I)
    p2_pattern = re.compile(r"\bP2\s+payoff\s*(?:=|:)\s*(-?\d+)\b", re.I)
    p1 = p1_pattern.search(text)
    p2 = p2_pattern.search(text)
    return (
        p1 is not None
        and p2 is not None
        and int(p1.group(1)) == int(target["p1_payoff"])
        and int(p2.group(1)) == int(target["p2_payoff"])
    )


def prediction_correct(text: str, row: dict[str, object]) -> bool:
    family = task_family(row)
    meta = metadata(row)
    if family in {"large_normal_form", "natural_language"}:
        rows = [str(value) for value in meta["rows"]]
        cols = [str(value) for value in meta["cols"]]
        return parse_profiles(text, rows, cols) == gold_profiles(row)
    if family == "mixed_2x2":
        return mixed_matches(parse_mixed_prediction(text, row), row)
    if family == "dominance":
        return dominance_matches(text, row)
    if family == "extensive_form":
        return extensive_matches(text, row)
    if family == "repeated_interaction":
        return repeated_matches(text, row)
    raise ValueError(f"unknown task_family: {family}")


def gold_summary(row: dict[str, object]) -> object:
    family = task_family(row)
    meta = metadata(row)
    if family in {"large_normal_form", "natural_language"}:
        return sorted(gold_profiles(row))
    if family == "mixed_2x2":
        return meta["mixed_nash_equilibrium"]
    if family == "dominance":
        elimination = meta["iterated_elimination"]
        if isinstance(elimination, dict):
            return {
                "remaining_rows": elimination.get("remaining_rows", []),
                "remaining_cols": elimination.get("remaining_cols", []),
            }
    if family == "extensive_form":
        return meta["subgame_perfect_equilibrium"]
    if family == "repeated_interaction":
        return meta["simulation"]
    return None


def score_suite(gold_rows: list[dict[str, object]], pred_rows: list[dict[str, object]]) -> dict[str, object]:
    predictions = {
        str(row.get("id")): str(row.get("prediction", ""))
        for row in pred_rows
    }
    correct = 0
    by_family: dict[str, dict[str, float | int]] = {}
    failed_examples: list[dict[str, object]] = []

    for gold_row in gold_rows:
        example_id = str(gold_row["id"])
        family = task_family(gold_row)
        prediction = predictions.get(example_id, "")
        is_correct = prediction_correct(prediction, gold_row)
        update_stats(by_family.setdefault(family, empty_stats()), is_correct)
        if is_correct:
            correct += 1
        else:
            failed_examples.append(
                {
                    "id": example_id,
                    "task_family": family,
                    "prompt": gold_row.get("prompt", ""),
                    "gold": gold_summary(gold_row),
                    "prediction": prediction,
                }
            )

    finalize_stats(by_family)
    total = len(gold_rows)
    return {
        "total_examples": total,
        "exact_match_accuracy": correct / total if total else 0.0,
        "num_correct": correct,
        "num_incorrect": total - correct,
        "accuracy_by_task_family": by_family,
        "failed_examples": failed_examples,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score broader GT-Bench suite predictions.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = score_suite(read_jsonl(args.gold), read_jsonl(args.pred))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
