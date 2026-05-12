from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

from game_theory_suite import DEFAULT_SUITE_FAMILIES
from score_predictions import gold_profiles, parse_prediction, read_jsonl


CANDIDATE_RUNS = [
    ("joint_base_canon_adv_suite", "Joint base canonical+adv+suite"),
    ("joint_adv_suite_retention", "Joint adv-state suite+retention"),
    ("joint_adv_targeted_retention", "Joint adv-state targeted+retention"),
    ("joint_base_full_targeted", "Joint base full targeted"),
    ("joint_followup_retention", "Joint follow-up retention"),
    ("joint_followup_suite", "Joint follow-up suite"),
]

EVALUATIONS = {
    "canonical": "score_predictions",
    "confirmation": "score_predictions",
    "stress": "score_predictions",
    "robustness": "score_robustness",
    "suite": "score_suite",
}


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def report_path(run: str, evaluation: str) -> Path:
    if evaluation == "suite":
        return Path(f"reports/suite_{run}_report.json")
    return Path(f"reports/{run}_{evaluation}_report.json")


def prediction_path(run: str, evaluation: str) -> Path:
    if evaluation == "suite":
        return Path(f"predictions/suite_{run}.jsonl")
    return Path(f"predictions/{run}_{evaluation}.jsonl")


def summarize_report(report: dict[str, Any], kind: str) -> dict[str, Any]:
    summary = {
        "total_examples": report["total_examples"],
        "exact_match_accuracy": report["exact_match_accuracy"],
        "num_correct": report["num_correct"],
        "num_incorrect": report["num_incorrect"],
    }
    if kind == "score_suite":
        summary["accuracy_by_task_family"] = report.get("accuracy_by_task_family", {})
        summary["accuracy_by_difficulty"] = report.get("accuracy_by_difficulty", {})
    if kind == "score_robustness":
        summary["accuracy_by_prompt_variant"] = report.get("accuracy_by_prompt_variant", {})
    if kind == "score_predictions":
        summary["accuracy_by_number_of_equilibria"] = report.get(
            "accuracy_by_number_of_equilibria",
            {},
        )
    return summary


def load_eval(
    run: str,
    evaluation: str,
    previous: dict[str, Any] | None,
) -> dict[str, Any] | None:
    path = report_path(run, evaluation)
    if path.exists():
        summary = summarize_report(load_json(path), EVALUATIONS[evaluation])
        summary["source_report"] = str(path)
        return summary
    if previous:
        return (
            previous.get("runs", {})
            .get(run, {})
            .get("evaluations", {})
            .get(evaluation)
        )
    return None


def complete_evaluations(evaluations: dict[str, Any]) -> bool:
    return all(name in evaluations for name in ("canonical", "robustness", "suite"))


def select_checkpoint(runs: dict[str, Any]) -> dict[str, Any]:
    complete = [
        (name, row)
        for name, row in runs.items()
        if complete_evaluations(row.get("evaluations", {}))
    ]
    eligible = [
        (name, row)
        for name, row in complete
        if row["evaluations"]["canonical"]["exact_match_accuracy"] >= 0.99
        and row["evaluations"]["robustness"]["exact_match_accuracy"] >= 0.95
    ]
    threshold = "canonical>=99.0 and robustness>=95.0"
    if not eligible:
        eligible = [
            (name, row)
            for name, row in complete
            if row["evaluations"]["canonical"]["exact_match_accuracy"] >= 0.95
        ]
        threshold = "fallback canonical>=95.0"
    if not eligible:
        return {"status": "no_complete_eligible_checkpoint", "rule": threshold}
    name, row = max(
        eligible,
        key=lambda item: item[1]["evaluations"]["suite"]["exact_match_accuracy"],
    )
    return {
        "status": "selected",
        "rule": threshold,
        "run": name,
        "label": row["label"],
        "canonical_accuracy": row["evaluations"]["canonical"]["exact_match_accuracy"],
        "robustness_accuracy": row["evaluations"]["robustness"]["exact_match_accuracy"],
        "suite_accuracy": row["evaluations"]["suite"]["exact_match_accuracy"],
    }


def canonical_failure_diagnostics(gold_path: Path, pred_path: Path) -> dict[str, Any]:
    if not gold_path.exists() or not pred_path.exists():
        return {}
    gold_rows = read_jsonl(gold_path)
    pred_rows = {str(row.get("id")): str(row.get("prediction", "")) for row in read_jsonl(pred_path)}
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for row in gold_rows:
        gold = gold_profiles(row)
        parsed = parse_prediction(pred_rows.get(str(row["id"]), ""))
        if parsed == gold:
            continue
        if parsed is None:
            label = "parser_failure"
        else:
            false_positive_profiles = parsed - gold
            false_negative_profiles = gold - parsed
            if not gold and parsed:
                label = "zero_equilibrium_false_positive"
            elif false_positive_profiles and not false_negative_profiles:
                label = "false_positive_profile"
            elif false_negative_profiles and not false_positive_profiles:
                label = "false_negative_profile"
            else:
                label = "mixed_false_positive_false_negative"
        if len(gold) >= 2:
            counts["tie_heavy_error"] += 1
        counts[label] += 1
        if len(examples) < 8:
            examples.append(
                {
                    "id": row["id"],
                    "type": label,
                    "gold": sorted(gold),
                    "parsed": None if parsed is None else sorted(parsed),
                }
            )
    return {"counts": dict(counts), "examples": examples}


def suite_failure_diagnostics(report: dict[str, Any]) -> dict[str, Any]:
    failed = report.get("failed_examples", [])
    by_family: dict[str, int] = defaultdict(int)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in failed:
        family = str(row.get("task_family", "unknown"))
        by_family[family] += 1
        if len(examples[family]) < 2:
            examples[family].append(
                {
                    "id": row.get("id"),
                    "gold": row.get("gold"),
                    "prediction": str(row.get("prediction", ""))[:300],
                }
            )
    return {
        "failed_by_family": dict(sorted(by_family.items())),
        "examples_by_family": dict(examples),
        "accuracy_by_difficulty": report.get("accuracy_by_difficulty", {}),
    }


def maybe_selected_diagnostics(selection: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    if selection.get("status") != "selected":
        return previous.get("selected_diagnostics", {}) if previous else {}
    run = str(selection["run"])
    diagnostics: dict[str, Any] = {}
    diagnostics["canonical"] = canonical_failure_diagnostics(
        Path("data/test.jsonl"),
        prediction_path(run, "canonical"),
    )
    suite_report = report_path(run, "suite")
    if suite_report.exists():
        diagnostics["suite"] = suite_failure_diagnostics(load_json(suite_report))
    elif previous:
        diagnostics["suite"] = previous.get("selected_diagnostics", {}).get("suite", {})
    return diagnostics


def load_external_baselines(previous: dict[str, Any] | None) -> dict[str, Any]:
    availability_path = Path("reports/model_availability.json")
    availability = load_json(availability_path) if availability_path.exists() else {}
    selected = availability.get("selected_models", [])
    baselines: dict[str, Any] = {}
    for item in selected:
        model = str(item["model"])
        slug = slug_model(model)
        canonical = Path(f"reports/base_{slug}_canonical_report.json")
        suite = Path(f"reports/suite_base_{slug}_report.json")
        row = {
            "model": model,
            "reason": item.get("reason", ""),
            "canonical": summarize_report(load_json(canonical), "score_predictions") if canonical.exists() else None,
            "suite": summarize_report(load_json(suite), "score_suite") if suite.exists() else None,
        }
        if row["canonical"] or row["suite"]:
            baselines[slug] = row
        elif previous and slug in previous.get("external_baselines", {}):
            baselines[slug] = previous["external_baselines"][slug]
    return baselines


def slug_model(model: str) -> str:
    return (
        model.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "")
        .replace(":", "_")
    )


def summarize_seed_runs(runs: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for name, row in runs.items():
        for seed in (42, 1009, 2027):
            suffix = f"_seed{seed}"
            if name.endswith(suffix):
                grouped[name.removesuffix(suffix)][seed] = row
    output: dict[str, Any] = {}
    for base_name, rows_by_seed in grouped.items():
        if 42 not in rows_by_seed and base_name in runs:
            rows_by_seed[42] = runs[base_name]
        rows = [rows_by_seed[seed] for seed in sorted(rows_by_seed)]
        suite_acc = [
            row["evaluations"]["suite"]["exact_match_accuracy"]
            for row in rows
            if "suite" in row["evaluations"]
        ]
        canonical_acc = [
            row["evaluations"]["canonical"]["exact_match_accuracy"]
            for row in rows
            if "canonical" in row["evaluations"]
        ]
        family_values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            suite = row["evaluations"].get("suite")
            if suite:
                for family, stats in suite.get("accuracy_by_task_family", {}).items():
                    family_values[family].append(float(stats["accuracy"]))
        output[base_name] = {
            "num_runs": len(rows),
            "suite_mean": mean(suite_acc) if suite_acc else None,
            "suite_seed_sd": pstdev(suite_acc) if len(suite_acc) > 1 else 0.0,
            "canonical_mean": mean(canonical_acc) if canonical_acc else None,
            "canonical_seed_sd": pstdev(canonical_acc) if len(canonical_acc) > 1 else 0.0,
            "suite_family_seed_sd": {
                family: pstdev(values) if len(values) > 1 else 0.0
                for family, values in family_values.items()
            },
            "per_seed": {
                str(seed): {
                    evaluation: row["evaluations"][evaluation]["exact_match_accuracy"]
                    for evaluation in ("canonical", "suite")
                    if evaluation in row["evaluations"]
                }
                for seed, row in sorted(rows_by_seed.items())
            },
        }
    return output


def build_summary(previous: dict[str, Any] | None = None) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    run_names = list(CANDIDATE_RUNS)
    for base_name, label in CANDIDATE_RUNS:
        for seed in (42, 1009, 2027):
            seeded_name = f"{base_name}_seed{seed}"
            if any(report_path(seeded_name, evaluation).exists() for evaluation in EVALUATIONS):
                run_names.append((seeded_name, f"{label} seed {seed}"))
    for name, label in run_names:
        evaluations = {}
        for evaluation in EVALUATIONS:
            loaded = load_eval(name, evaluation, previous)
            if loaded:
                evaluations[evaluation] = loaded
        if evaluations or (previous and name in previous.get("runs", {})):
            previous_row = previous.get("runs", {}).get(name, {}) if previous else {}
            runs[name] = {
                "label": label or previous_row.get("label", name),
                "evaluations": evaluations or previous_row.get("evaluations", {}),
            }
    selection = select_checkpoint(runs)
    if selection.get("status") != "selected" and previous and previous.get("selection"):
        selection = previous["selection"]
    return {
        "status": "complete" if selection.get("status") == "selected" else "incomplete",
        "runs": runs,
        "selection": selection,
        "seed_summary": summarize_seed_runs(runs),
        "external_baselines": load_external_baselines(previous),
        "selected_diagnostics": maybe_selected_diagnostics(selection, previous),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# GT-Bench Multitask Results", ""]
    selection = summary.get("selection", {})
    if selection.get("status") == "selected":
        lines.append(
            "Selected checkpoint: "
            f"`{selection['run']}` ({pct(float(selection['suite_accuracy']))} suite, "
            f"{pct(float(selection['canonical_accuracy']))} canonical, "
            f"{pct(float(selection['robustness_accuracy']))} robustness)."
        )
    else:
        lines.append("Selected checkpoint: pending complete multitask reports.")
    lines.extend(["", "## Candidate Sweep", ""])
    lines.append("| Run | Canonical | Robustness | Suite |")
    lines.append("| --- | ---: | ---: | ---: |")
    for name, row in summary.get("runs", {}).items():
        evaluations = row.get("evaluations", {})
        lines.append(
            f"| `{name}` | {maybe_pct(evaluations, 'canonical')} | "
            f"{maybe_pct(evaluations, 'robustness')} | {maybe_pct(evaluations, 'suite')} |"
        )
    if summary.get("seed_summary"):
        lines.extend(["", "## Multi-Seed Summary", ""])
        lines.append("| Recipe | Runs | Suite mean | Suite seed SD | Canonical mean | Canonical seed SD |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for name, row in summary["seed_summary"].items():
            lines.append(
                f"| `{name}` | {row['num_runs']} | {optional_pct(row['suite_mean'])} | "
                f"{optional_pp(row['suite_seed_sd'])} | {optional_pct(row['canonical_mean'])} | "
                f"{optional_pp(row['canonical_seed_sd'])} |"
            )
        lines.extend(["", "| Recipe | Family | Suite seed SD |"])
        lines.append("| --- | --- | ---: |")
        for name, row in summary["seed_summary"].items():
            for family, sd in sorted(row.get("suite_family_seed_sd", {}).items()):
                lines.append(f"| `{name}` | `{family}` | {optional_pp(sd)} |")
    if summary.get("external_baselines"):
        lines.extend(["", "## External Base Models", ""])
        lines.append("| Model | Canonical | Suite |")
        lines.append("| --- | ---: | ---: |")
        for row in summary["external_baselines"].values():
            canonical = row.get("canonical")
            suite = row.get("suite")
            lines.append(
                f"| `{row['model']}` | {optional_pct(canonical and canonical['exact_match_accuracy'])} | "
                f"{optional_pct(suite and suite['exact_match_accuracy'])} |"
            )
    diagnostics = summary.get("selected_diagnostics", {})
    if diagnostics:
        lines.extend(["", "## Selected Failure Diagnostics", ""])
        canonical = diagnostics.get("canonical", {})
        if canonical.get("counts"):
            lines.append("Canonical failure counts: " + ", ".join(
                f"`{key}`={value}" for key, value in canonical["counts"].items()
            ) + ".")
        suite = diagnostics.get("suite", {})
        if suite.get("failed_by_family"):
            lines.append("Suite failures by family: " + ", ".join(
                f"`{key}`={value}" for key, value in suite["failed_by_family"].items()
            ) + ".")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_pct(evaluations: dict[str, Any], key: str) -> str:
    row = evaluations.get(key)
    return optional_pct(row and row.get("exact_match_accuracy"))


def optional_pct(value: object) -> str:
    return "n/a" if value is None else pct(float(value))


def optional_pp(value: object) -> str:
    return "n/a" if value is None or math.isnan(float(value)) else f"{100 * float(value):.2f} pp"


def write_figures(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_pareto(out_dir / "multitask_pareto.svg", summary)
    write_external(out_dir / "external_model_baselines.svg", summary)


def write_pareto(path: Path, summary: dict[str, Any]) -> None:
    points = []
    labels = {name: label for name, label in CANDIDATE_RUNS}
    for name, _label in CANDIDATE_RUNS:
        row = summary.get("runs", {}).get(name, {})
        evaluations = row.get("evaluations", {})
        if "canonical" in evaluations and "suite" in evaluations:
            points.append((name, evaluations["canonical"]["exact_match_accuracy"], evaluations["suite"]["exact_match_accuracy"]))
    width, height = 900, 500
    left, top, plot_w, plot_h = 90, 58, 740, 330
    xmin, xmax = 0.95, 1.0
    ymin, ymax = 0.80, 0.95

    def x_pos(value: float) -> float:
        return left + plot_w * (value - xmin) / (xmax - xmin)

    def y_pos(value: float) -> float:
        return top + plot_h * (ymax - value) / (ymax - ymin)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="30" font-family="Arial" font-size="20" font-weight="700">Retention vs Suite Accuracy</text>',
        '<text x="20" y="50" font-family="Arial" font-size="12" fill="#475569">Candidate multitask checkpoints on the fixed public suite split.</text>',
    ]
    for tick in (95, 96, 97, 98, 99, 100):
        x = x_pos(tick / 100)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{x:.1f}" y="{top + plot_h + 20}" text-anchor="middle" font-family="Arial" font-size="11">{tick}%</text>')
    for tick in (80, 85, 90, 95):
        y = y_pos(tick / 100)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{tick}%</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>')
    for name, canonical, suite in points:
        x = x_pos(canonical)
        y = y_pos(suite)
        selected = name == summary.get("selection", {}).get("run")
        color = "#111827" if selected else "#0072b2"
        label = "selected" if selected else labels.get(name, name).replace("Joint ", "")
        label = f"{label} ({pct(suite)})"
        anchor = "end" if canonical > 0.99 else "start"
        text_x = x - 10 if anchor == "end" else x + 10
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
        lines.append(f'<text x="{text_x:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}" font-family="Arial" font-size="11" fill="{color}">{label}</text>')
    lines.append('<text x="460" y="460" text-anchor="middle" font-family="Arial" font-size="12">Canonical accuracy</text>')
    lines.append('<text x="18" y="235" transform="rotate(-90 18 235)" text-anchor="middle" font-family="Arial" font-size="12">Suite accuracy</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_external(path: Path, summary: dict[str, Any]) -> None:
    rows = list(summary.get("external_baselines", {}).values())
    width, height = 760, 360
    left, top, plot_w, plot_h = 150, 50, 540, 220
    max_acc = max(
        [
            float(row.get(key, {}).get("exact_match_accuracy", 0.0))
            for row in rows
            for key in ("canonical", "suite")
            if row.get(key)
        ]
        or [0.0]
    )
    y_max = max(0.08, math.ceil(max_acc * 100) / 100)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="30" font-family="Arial" font-size="20" font-weight="700">External Base Model Accuracy</text>',
    ]
    for index in range(5):
        value = y_max * index / 4
        y = top + plot_h * (1 - value / y_max)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eeeeee"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{pct(value)}</text>')
    group_w = plot_w / max(len(rows), 1)
    colors = {"canonical": "#0072b2", "suite": "#d55e00"}
    for index, row in enumerate(rows):
        x0 = left + index * group_w + group_w * 0.25
        for offset, key in enumerate(("canonical", "suite")):
            report = row.get(key)
            acc = float(report.get("exact_match_accuracy", 0.0)) if report else 0.0
            h = plot_h * acc / y_max
            x = x0 + offset * 28
            y = top + plot_h - h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="22" height="{h:.1f}" fill="{colors[key]}"/>')
            lines.append(f'<text x="{x + 11:.1f}" y="{y - 4:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{pct(acc)}</text>')
        lines.append(f'<text x="{x0 + 18:.1f}" y="{height - 48}" text-anchor="middle" font-family="Arial" font-size="11">{row["model"].split("/")[-1]}</text>')
    lines.append('<rect x="150" y="320" width="12" height="12" fill="#0072b2"/><text x="168" y="331" font-family="Arial" font-size="12">Canonical</text>')
    lines.append('<rect x="260" y="320" width="12" height="12" fill="#d55e00"/><text x="278" y="331" font-family="Arial" font-size="12">Suite</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paper-grade multitask GT-Bench results.")
    parser.add_argument("--out-json", type=Path, default=Path("reports/multitask_results.json"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/multitask_results.md"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    previous = load_json(args.out_json) if args.out_json.exists() else None
    summary = build_summary(previous)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, summary)
    write_figures(args.figures_dir, summary)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
