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


DEFAULT_MODEL_REPORTS = [
    (
        "base_qwen36_27b",
        "Base Qwen/Qwen3.6-27B",
        Path("reports/suite_base_qwen36_27b_report.json"),
    ),
    (
        "pure_2x2_sft_5000",
        "2x2 SFT transfer",
        Path("reports/suite_qwen36_27b_sft_5000_report.json"),
    ),
    (
        "pure_2x2_prompt_adv500",
        "2x2 + prompt-adversarial SFT transfer",
        Path("reports/suite_qwen36_27b_sft_5000_plus_prompt_adv500_report.json"),
    ),
    (
        "suite_sft_1200",
        "Suite SFT",
        Path("reports/suite_qwen36_27b_suite_sft_1200_report.json"),
    ),
    (
        "joint_base_canon_adv_suite",
        "Joint base canonical+adv+suite",
        Path("reports/suite_joint_base_canon_adv_suite_report.json"),
    ),
    (
        "joint_adv_suite_retention",
        "Joint adv-state suite+retention",
        Path("reports/suite_joint_adv_suite_retention_report.json"),
    ),
    (
        "joint_adv_targeted_retention",
        "Joint adv-state targeted+retention",
        Path("reports/suite_joint_adv_targeted_retention_report.json"),
    ),
    (
        "joint_base_full_targeted",
        "Joint base full targeted",
        Path("reports/suite_joint_base_full_targeted_report.json"),
    ),
    (
        "joint_followup_retention",
        "Joint follow-up retention",
        Path("reports/suite_joint_followup_retention_report.json"),
    ),
    (
        "joint_followup_suite",
        "Joint follow-up suite",
        Path("reports/suite_joint_followup_suite_report.json"),
    ),
    (
        "base_qwen3_8b",
        "Base Qwen/Qwen3-8B",
        Path("reports/suite_base_qwen_qwen3_8b_report.json"),
    ),
    (
        "base_qwen3_30b_a3b",
        "Base Qwen/Qwen3-30B-A3B",
        Path("reports/suite_base_qwen_qwen3_30b_a3b_report.json"),
    ),
]

DEFAULT_RETENTION_REPORTS = [
    (
        "suite_sft_1200_on_canonical",
        "Suite SFT on canonical 2x2",
        Path("reports/qwen36_27b_suite_sft_1200_on_canonical_report.json"),
    ),
    (
        "joint_base_canon_adv_suite_on_canonical",
        "Joint base canonical+adv+suite on canonical 2x2",
        Path("reports/joint_base_canon_adv_suite_canonical_report.json"),
    ),
    (
        "joint_adv_suite_retention_on_canonical",
        "Joint adv-state suite+retention on canonical 2x2",
        Path("reports/joint_adv_suite_retention_canonical_report.json"),
    ),
    (
        "joint_adv_targeted_retention_on_canonical",
        "Joint adv-state targeted+retention on canonical 2x2",
        Path("reports/joint_adv_targeted_retention_canonical_report.json"),
    ),
    (
        "joint_base_full_targeted_on_canonical",
        "Joint base full targeted on canonical 2x2",
        Path("reports/joint_base_full_targeted_canonical_report.json"),
    ),
    (
        "joint_followup_retention_on_canonical",
        "Joint follow-up retention on canonical 2x2",
        Path("reports/joint_followup_retention_canonical_report.json"),
    ),
    (
        "joint_followup_suite_on_canonical",
        "Joint follow-up suite on canonical 2x2",
        Path("reports/joint_followup_suite_canonical_report.json"),
    ),
]


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
    return {
        "total_examples": report["total_examples"],
        "exact_match_accuracy": report["exact_match_accuracy"],
        "num_correct": report["num_correct"],
        "num_incorrect": report["num_incorrect"],
        "accuracy_by_task_family": report["accuracy_by_task_family"],
    }


def summarize_model_report(label: str, report: dict[str, Any], source_report: Path) -> dict[str, Any]:
    summary = summarize_report(report)
    summary["label"] = label
    summary["source_report"] = str(source_report)
    return summary


def summarize_retention_report(label: str, report: dict[str, Any], source_report: Path) -> dict[str, Any]:
    return {
        "label": label,
        "source_report": str(source_report),
        "total_examples": report["total_examples"],
        "exact_match_accuracy": report["exact_match_accuracy"],
        "num_correct": report["num_correct"],
        "num_incorrect": report["num_incorrect"],
        "accuracy_by_number_of_equilibria": report.get("accuracy_by_number_of_equilibria", {}),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_named_reports(
    default_reports: Sequence[tuple[str, str, Path]],
    previous: dict[str, Any] | None,
    suite_sha256: str | None,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for name, label, path in default_reports:
        if path.exists():
            reports[name] = summarize_model_report(label, load_json(path), path)
            continue
        if previous and previous.get("suite_sha256") == suite_sha256:
            previous_reports = previous.get("model_evaluations")
            if isinstance(previous_reports, dict) and name in previous_reports:
                reports[name] = previous_reports[name]
    return reports


def load_retention_reports(
    default_reports: Sequence[tuple[str, str, Path]],
    previous: dict[str, Any] | None,
    suite_sha256: str | None,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for name, label, path in default_reports:
        if path.exists():
            reports[name] = summarize_retention_report(label, load_json(path), path)
            continue
        if previous and previous.get("suite_sha256") == suite_sha256:
            previous_reports = previous.get("canonical_retention")
            if isinstance(previous_reports, dict) and name in previous_reports:
                reports[name] = previous_reports[name]
    return reports


def is_public_suite_split(gold_path: Path) -> bool:
    public_path = Path("data/suite/test.jsonl")
    if gold_path == public_path:
        return True
    try:
        return gold_path.resolve() == public_path.resolve()
    except OSError:
        return False


def write_suite_figure(path: Path, summary: dict[str, Any]) -> None:
    baselines = summary["baselines"]
    models = summary.get("model_evaluations", {})
    families = list(DEFAULT_SUITE_FAMILIES)
    series: list[tuple[str, str, dict[str, Any]]] = []
    for name, label in (
        ("always_none", "Always none"),
        ("random", "Random"),
        ("most_common_by_family", "Most common"),
    ):
        if name in baselines:
            series.append((name, label, baselines[name]))
    for name, _label, _path in DEFAULT_MODEL_REPORTS:
        if name in models:
            series.append((name, str(models[name].get("label", name)), models[name]))
    if "oracle" in baselines:
        series.append(("oracle", "Oracle", baselines["oracle"]))
    width = 1280
    legend_rows = max(1, (len(series) + 3) // 4)
    height = 440 + legend_rows * 28
    margin_left = 130
    margin_bottom = 80 + legend_rows * 28
    chart_width = width - margin_left - 35
    chart_height = height - 80 - margin_bottom
    group_width = chart_width / len(families)
    bar_width = min(20, group_width / max(len(series), 1) - 3)
    colors = {
        "always_none": "#8c8c8c",
        "random": "#d55e00",
        "most_common_by_family": "#0072b2",
        "base_qwen36_27b": "#cc79a7",
        "pure_2x2_sft_5000": "#56b4e9",
        "suite_sft_1200": "#009e73",
        "pure_2x2_prompt_adv500": "#f0e442",
        "oracle": "#222222",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="32" font-family="Arial" font-size="20" font-weight="700">GT-Bench Suite Accuracy By Task Family</text>',
    ]
    for tick in range(0, 101, 25):
        y = 60 + chart_height * (1 - tick / 100)
        lines.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - 30}" y2="{y:.1f}" stroke="#e0e0e0"/>')
        lines.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{tick}%</text>')
    for family_index, task_family in enumerate(families):
        x0 = margin_left + family_index * group_width + group_width * 0.18
        for name_index, (name, _label, row) in enumerate(series):
            family_stats = row["accuracy_by_task_family"].get(task_family, {})
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
            f'<text x="{margin_left + family_index * group_width + group_width / 2:.1f}" y="{height - 92}" '
            f'text-anchor="middle" font-family="Arial" font-size="11">{label}</text>'
        )
    legend_x = margin_left
    for index, (name, label, _row) in enumerate(series):
        x = legend_x + (index % 4) * 280
        y = height - (28 * legend_rows) + (index // 4) * 22
        lines.append(f'<rect x="{x}" y="{y}" width="12" height="12" fill="{colors.get(name, "#555")}"/>')
        lines.append(
            f'<text x="{x + 18}" y="{y + 10}" font-family="Arial" font-size="12">{label}</text>'
        )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GT-Bench Suite Results",
        "",
        (
            "Status: deterministic baselines and Tinker model evaluations are complete for "
            "the listed checkpoints."
            if summary.get("model_evaluations")
            else "Status: deterministic suite smoke baselines are complete; no model reports are attached."
        ),
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
    if summary.get("model_evaluations"):
        lines.extend(["", "## Model Evaluations", ""])
        lines.append("| Checkpoint | Accuracy | Correct | Incorrect |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in summary["model_evaluations"].values():
            lines.append(
                f"| {row['label']} | {pct(float(row['exact_match_accuracy']))} | "
                f"{row['num_correct']} | {row['num_incorrect']} |"
            )
    if summary.get("canonical_retention"):
        lines.extend(["", "## Canonical 2x2 Retention", ""])
        lines.append("| Checkpoint | Accuracy | Correct | Incorrect |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in summary["canonical_retention"].values():
            lines.append(
                f"| {row['label']} | {pct(float(row['exact_match_accuracy']))} | "
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
    if summary.get("model_evaluations"):
        lines.extend(["## Model Accuracy By Task Family", ""])
        lines.append("| Checkpoint | " + " | ".join(f"`{task_family}`" for task_family in DEFAULT_SUITE_FAMILIES) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in DEFAULT_SUITE_FAMILIES) + " |")
        for row in summary["model_evaluations"].values():
            cells = [
                pct(float(row["accuracy_by_task_family"].get(task_family, {}).get("accuracy", 0.0)))
                for task_family in DEFAULT_SUITE_FAMILIES
            ]
            lines.append(f"| {row['label']} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_suite_summary(
    gold_path: Path,
    train_path: Path | None,
    seed: int,
    pred_dir: Path | None,
    previous_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gold_rows = read_jsonl(gold_path)
    train_rows = read_jsonl(train_path) if train_path and train_path.exists() else []
    suite_sha256 = sha256_file(gold_path) if gold_path.exists() else None
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

    if is_public_suite_split(gold_path):
        model_evaluations = load_named_reports(DEFAULT_MODEL_REPORTS, previous_summary, suite_sha256)
        canonical_retention = load_retention_reports(DEFAULT_RETENTION_REPORTS, previous_summary, suite_sha256)
    else:
        model_evaluations = {}
        canonical_retention = {}

    return {
        "status": "model_evaluations_complete" if model_evaluations else "baseline_only",
        "suite_path": str(gold_path),
        "suite_sha256": suite_sha256,
        "train_path": str(train_path) if train_path else None,
        "train_sha256": sha256_file(train_path) if train_path and train_path.exists() else None,
        "baselines": baselines,
        "model_evaluations": model_evaluations,
        "canonical_retention": canonical_retention,
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
    previous_summary = load_json(args.out_json) if args.out_json.exists() else None
    summary = build_suite_summary(args.gold, args.train, args.seed, args.pred_dir, previous_summary)
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
