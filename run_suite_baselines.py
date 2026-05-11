from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from game_theory_suite import DEFAULT_SUITE_FAMILIES, ordered_profiles, to_suite_chat_row
from generate_dataset import write_jsonl
from score_predictions import read_jsonl
from score_suite import score_suite
from tinker_common import sha256_file


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def family(row: dict[str, object]) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"row {row.get('id')} is missing metadata")
    return str(metadata["task_family"])


def metadata(row: dict[str, object]) -> dict[str, Any]:
    value = row.get("metadata")
    if not isinstance(value, dict):
        raise ValueError(f"row {row.get('id')} is missing metadata")
    return value


def rows_cols(row: dict[str, object]) -> tuple[list[str], list[str]]:
    meta = metadata(row)
    return [str(value) for value in meta.get("rows", [])], [str(value) for value in meta.get("cols", [])]


def random_profile_answer(row: dict[str, object], rng: random.Random) -> str:
    rows, cols = rows_cols(row)
    profiles = ordered_profiles(rows, cols)
    selected = [profile for profile in profiles if rng.random() < 0.25]
    if not selected:
        return "NE=none."
    return "NE=" + ", ".join(f"({row_name}, {col_name})" for row_name, col_name in selected) + "."


def random_mixed_answer(rng: random.Random) -> str:
    p1_u = rng.choice([0.25, 0.5, 0.75])
    p2_l = rng.choice([0.25, 0.5, 0.75])
    return f"P1: U={p1_u}, D={1 - p1_u}; P2: L={p2_l}, R={1 - p2_l}."


def random_dominance_answer(row: dict[str, object], rng: random.Random) -> str:
    rows, cols = rows_cols(row)
    kept_rows = [row_name for row_name in rows if rng.random() < 0.5] or [rng.choice(rows)]
    kept_cols = [col_name for col_name in cols if rng.random() < 0.5] or [rng.choice(cols)]
    return (
        f"Remaining P1 strategies: {', '.join(kept_rows)}; "
        f"Remaining P2 strategies: {', '.join(kept_cols)}."
    )


def random_extensive_answer(rng: random.Random) -> str:
    p1 = rng.choice(["Left", "Right"])
    left = rng.choice(["A", "B"])
    right = rng.choice(["A", "B"])
    path = f"{p1}-{left if p1 == 'Left' else right}"
    return f"P1={p1}; P2 after Left={left}; P2 after Right={right}; Path={path}."


def random_repeated_answer(row: dict[str, object], rng: random.Random) -> str:
    horizon = int(metadata(row).get("horizon", 5))
    max_payoff = horizon * 5
    if metadata(row).get("repeated_task") == "best_response":
        strategy = rng.choice(["AlwaysC", "AlwaysD", "TitForTat", "GrimTrigger"])
        return f"Best P1 strategy: {strategy}; P1 payoff: {rng.randint(0, max_payoff)}."
    return f"P1 payoff: {rng.randint(0, max_payoff)}; P2 payoff: {rng.randint(0, max_payoff)}."


def baseline_prediction(
    row: dict[str, object],
    name: str,
    rng: random.Random,
    most_common: dict[str, str] | None = None,
) -> str:
    task = family(row)
    if name == "oracle":
        return str(row["answer"])
    if name == "most_common_by_family" and most_common is not None:
        return most_common.get(task, "")
    if name == "always_none":
        if task in {"large_normal_form", "natural_language"}:
            return "NE=none."
        if task == "mixed_2x2":
            return "P1: U=1/2, D=1/2; P2: L=1/2, R=1/2."
        if task == "dominance":
            rows, cols = rows_cols(row)
            return (
                f"Remaining P1 strategies: {rows[0]}; "
                f"Remaining P2 strategies: {cols[0]}."
            )
        if task == "extensive_form":
            return "P1=Left; P2 after Left=A; P2 after Right=A; Path=Left-A."
        if task == "repeated_interaction":
            if metadata(row).get("repeated_task") == "best_response":
                return "Best P1 strategy: AlwaysC; P1 payoff: 0."
            return "P1 payoff: 0; P2 payoff: 0."
    if name == "random":
        if task in {"large_normal_form", "natural_language"}:
            return random_profile_answer(row, rng)
        if task == "mixed_2x2":
            return random_mixed_answer(rng)
        if task == "dominance":
            return random_dominance_answer(row, rng)
        if task == "extensive_form":
            return random_extensive_answer(rng)
        if task == "repeated_interaction":
            return random_repeated_answer(row, rng)
    raise ValueError(f"unknown baseline {name!r} for task {task!r}")


def most_common_answers(train_rows: list[dict[str, object]]) -> dict[str, str]:
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train_rows:
        by_family[family(row)][str(row["answer"])] += 1
    return {
        task_family: counter.most_common(1)[0][0]
        for task_family, counter in by_family.items()
        if counter
    }


def summarize_report(report: dict[str, Any], prediction_path: Path | None = None) -> dict[str, Any]:
    summary = {
        "total_examples": report["total_examples"],
        "exact_match_accuracy": report["exact_match_accuracy"],
        "num_correct": report["num_correct"],
        "num_incorrect": report["num_incorrect"],
        "accuracy_by_task_family": report["accuracy_by_task_family"],
        "accuracy_by_difficulty": report.get("accuracy_by_difficulty", {}),
        "failed_examples_preview": report.get("failed_examples", [])[:10],
    }
    if prediction_path is not None:
        summary["prediction_path"] = str(prediction_path)
    return summary


def write_suite_figure(path: Path, summary: dict[str, Any]) -> None:
    baselines = summary["baselines"]
    families = list(DEFAULT_SUITE_FAMILIES)
    names = [name for name in ("always_none", "random", "most_common_by_family", "oracle") if name in baselines]
    width = 980
    height = 430
    margin_left = 130
    margin_bottom = 80
    chart_width = width - margin_left - 30
    chart_height = height - 80 - margin_bottom
    group_width = chart_width / len(families)
    bar_width = min(26, group_width / max(len(names), 1) - 4)
    colors = {
        "always_none": "#8c8c8c",
        "random": "#d55e00",
        "most_common_by_family": "#0072b2",
        "oracle": "#009e73",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="32" font-family="Arial" font-size="20" font-weight="700">Suite Smoke Baselines By Task Family</text>',
    ]
    for tick in range(0, 101, 25):
        y = 60 + chart_height * (1 - tick / 100)
        lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - 30}" y2="{y:.1f}" stroke="#e0e0e0"/>')
        lines.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{tick}%</text>')
    for family_index, task_family in enumerate(families):
        x0 = margin_left + family_index * group_width + group_width * 0.18
        for name_index, name in enumerate(names):
            family_stats = baselines[name]["accuracy_by_task_family"].get(task_family, {})
            acc = float(family_stats.get("accuracy", 0.0))
            bar_height = chart_height * acc
            x = x0 + name_index * (bar_width + 4)
            y = 60 + chart_height - bar_height
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
                f'fill="{colors.get(name, "#555")}"/>'
            )
        label = task_family.replace("_", " ")
        lines.append(
            f'<text x="{margin_left + family_index * group_width + group_width / 2:.1f}" y="{height - 38}" '
            f'text-anchor="middle" font-family="Arial" font-size="11">{label}</text>'
        )
    legend_x = margin_left
    for index, name in enumerate(names):
        x = legend_x + index * 190
        lines.append(f'<rect x="{x}" y="{height - 22}" width="12" height="12" fill="{colors.get(name, "#555")}"/>')
        lines.append(
            f'<text x="{x + 18}" y="{height - 12}" font-family="Arial" font-size="12">{name.replace("_", " ")}</text>'
        )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GT-Bench Suite Results",
        "",
        "Status: local deterministic suite smoke baselines are complete; Tinker model evaluations are pending.",
        "",
        f"Suite file: `{summary['suite_path']}`",
    ]
    if summary.get("suite_sha256"):
        lines.append(f"Suite SHA-256: `{summary['suite_sha256']}`")
    lines.extend(["", "## Baselines", ""])
    lines.append("| Baseline | Accuracy | Correct | Incorrect |")
    lines.append("| --- | ---: | ---: | ---: |")
    for name, row in summary["baselines"].items():
        lines.append(
            f"| `{name}` | {pct(float(row['exact_match_accuracy']))} | "
            f"{row['num_correct']} | {row['num_incorrect']} |"
        )
    lines.extend(["", "## Accuracy By Task Family", ""])
    for name, row in summary["baselines"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Task family | Accuracy | Correct | Total |")
        lines.append("| --- | ---: | ---: | ---: |")
        for task_family, stats in row["accuracy_by_task_family"].items():
            lines.append(
                f"| `{task_family}` | {pct(float(stats['accuracy']))} | "
                f"{stats['correct']} | {stats['total']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Pending Model Evaluations",
            "",
            "- base `Qwen/Qwen3.6-27B` on the suite test split",
            "- current 2x2 pure-equilibrium SFT checkpoint on the suite test split",
            "- full-suite SFT checkpoint",
            "- optional adversarial full-suite checkpoint",
            "",
            "No model-result claim should be made for the broader suite until these reports exist.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_suite_summary(
    gold_path: Path,
    train_path: Path | None,
    seed: int,
    pred_dir: Path | None,
) -> dict[str, Any]:
    gold_rows = read_jsonl(gold_path)
    train_rows = read_jsonl(train_path) if train_path and train_path.exists() else []
    common = most_common_answers(train_rows) if train_rows else None
    baseline_names = ["always_none", "random", "oracle"]
    if common:
        baseline_names.insert(2, "most_common_by_family")

    baselines: dict[str, Any] = {}
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)
    for name in baseline_names:
        rng = random.Random(seed)
        predictions = [
            {
                "id": row["id"],
                "prediction": baseline_prediction(row, name, rng, common),
            }
            for row in gold_rows
        ]
        prediction_path = None
        if pred_dir is not None:
            prediction_path = pred_dir / f"{name}.jsonl"
            write_jsonl(prediction_path, predictions)
        report = score_suite(gold_rows, predictions)
        baselines[name] = summarize_report(report, prediction_path)

    return {
        "status": "pending_model_evaluations",
        "suite_path": str(gold_path),
        "suite_sha256": sha256_file(gold_path) if gold_path.exists() else None,
        "train_path": str(train_path) if train_path else None,
        "train_sha256": sha256_file(train_path) if train_path and train_path.exists() else None,
        "baselines": baselines,
        "model_evaluations": {
            "base_model": {"status": "pending"},
            "pure_2x2_sft": {"status": "pending"},
            "suite_sft": {"status": "pending"},
            "suite_adversarial_sft": {"status": "pending"},
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local GT-Bench suite baselines.")
    parser.add_argument("--gold", type=Path, default=Path("data/suite/test.jsonl"))
    parser.add_argument("--train", type=Path, default=Path("data/suite/train.jsonl"))
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--pred-dir", type=Path, default=Path("predictions/suite_baselines"))
    parser.add_argument("--out-json", type=Path, default=Path("reports/suite_results.json"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/suite_results.md"))
    parser.add_argument("--figure", type=Path, default=Path("reports/figures/suite_smoke_accuracy.svg"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_suite_summary(args.gold, args.train, args.seed, args.pred_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, summary)
    write_suite_figure(args.figure, summary)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(f"wrote {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
