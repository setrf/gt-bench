from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from tinker_common import CONFIG_PATH, load_config, sha256_file


def read_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_named_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run values must look like name=path")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("run name cannot be empty")
    return name, Path(path)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_summary(config: dict[str, Any], baseline_path: Path, runs: list[tuple[str, Path]]) -> dict[str, Any]:
    baseline = read_report(baseline_path)
    run_summaries = []
    base_acc = float(baseline["exact_match_accuracy"])

    for name, path in runs:
        report = read_report(path)
        acc = float(report["exact_match_accuracy"])
        run_summaries.append(
            {
                "name": name,
                "report_path": str(path),
                "exact_match_accuracy": acc,
                "delta_vs_baseline": acc - base_acc,
                "num_correct": report["num_correct"],
                "num_incorrect": report["num_incorrect"],
                "accuracy_by_number_of_equilibria": report["accuracy_by_number_of_equilibria"],
                "failed_examples_preview": report.get("failed_examples", [])[:10],
            }
        )

    best = max(run_summaries, key=lambda row: row["exact_match_accuracy"], default=None)
    test_path = Path(config["data"]["test"])
    return {
        "model_id": config["model_id"],
        "baseline_report": str(baseline_path),
        "test_path": str(test_path),
        "test_sha256": sha256_file(test_path) if test_path.exists() else None,
        "baseline": {
            "exact_match_accuracy": base_acc,
            "num_correct": baseline["num_correct"],
            "num_incorrect": baseline["num_incorrect"],
            "accuracy_by_number_of_equilibria": baseline["accuracy_by_number_of_equilibria"],
            "failed_examples_preview": baseline.get("failed_examples", [])[:10],
        },
        "runs": run_summaries,
        "best_run": best,
    }


def add_confirmation_summary(
    summary: dict[str, Any],
    baseline_path: Path | None,
    runs: list[tuple[str, Path]],
    gold_path: Path | None,
) -> None:
    if baseline_path is None:
        return

    baseline = read_report(baseline_path)
    base_acc = float(baseline["exact_match_accuracy"])
    run_summaries = []
    for name, path in runs:
        report = read_report(path)
        acc = float(report["exact_match_accuracy"])
        run_summaries.append(
            {
                "name": name,
                "report_path": str(path),
                "exact_match_accuracy": acc,
                "delta_vs_baseline": acc - base_acc,
                "num_correct": report["num_correct"],
                "num_incorrect": report["num_incorrect"],
                "accuracy_by_number_of_equilibria": report["accuracy_by_number_of_equilibria"],
                "failed_examples_preview": report.get("failed_examples", [])[:10],
            }
        )

    summary["confirmation"] = {
        "gold_path": str(gold_path) if gold_path else None,
        "gold_sha256": sha256_file(gold_path) if gold_path and gold_path.exists() else None,
        "baseline_report": str(baseline_path),
        "baseline": {
            "exact_match_accuracy": base_acc,
            "num_correct": baseline["num_correct"],
            "num_incorrect": baseline["num_incorrect"],
            "accuracy_by_number_of_equilibria": baseline["accuracy_by_number_of_equilibria"],
            "failed_examples_preview": baseline.get("failed_examples", [])[:10],
        },
        "runs": run_summaries,
        "best_run": max(run_summaries, key=lambda row: row["exact_match_accuracy"], default=None),
    }


def append_accuracy_table(lines: list[str], baseline: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "| Run | Accuracy | Correct | Incorrect | Delta vs baseline |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.append(
        f"| baseline | {pct(float(baseline['exact_match_accuracy']))} | "
        f"{baseline['num_correct']} | {baseline['num_incorrect']} | 0.00 pp |"
    )
    for run in runs:
        delta_pp = float(run["delta_vs_baseline"]) * 100
        lines.append(
            f"| {run['name']} | {pct(float(run['exact_match_accuracy']))} | "
            f"{run['num_correct']} | {run['num_incorrect']} | {delta_pp:+.2f} pp |"
        )


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GT-Bench Qwen3.6-27B Results",
        "",
        f"Model: `{summary['model_id']}`",
        f"Test set: `{summary['test_path']}`",
    ]
    if summary.get("test_sha256"):
        lines.append(f"Test SHA-256: `{summary['test_sha256']}`")
    lines.extend(
        [
            "",
            "## Exact-Match Accuracy",
            "",
        ]
    )

    baseline = summary["baseline"]
    append_accuracy_table(lines, baseline, summary["runs"])

    lines.extend(["", "## Accuracy By Number Of Equilibria", ""])
    for label, report in [("baseline", baseline), *[(run["name"], run) for run in summary["runs"]]]:
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Equilibria | Total | Correct | Accuracy |")
        lines.append("| ---: | ---: | ---: | ---: |")
        buckets = report["accuracy_by_number_of_equilibria"]
        for count in sorted(buckets, key=lambda key: int(key)):
            bucket = buckets[count]
            lines.append(
                f"| {count} | {bucket['total']} | {bucket['correct']} | "
                f"{pct(float(bucket['accuracy']))} |"
            )
        lines.append("")

    confirmation = summary.get("confirmation")
    if confirmation:
        lines.extend(["## Independent Confirmation", ""])
        if confirmation.get("gold_path"):
            lines.append(f"Confirmation set: `{confirmation['gold_path']}`")
        if confirmation.get("gold_sha256"):
            lines.append(f"Confirmation SHA-256: `{confirmation['gold_sha256']}`")
        lines.append("")
        append_accuracy_table(lines, confirmation["baseline"], confirmation["runs"])
        lines.append("")

    best = summary.get("best_run")
    if best:
        lines.extend(
            [
                "## Takeaway",
                "",
                f"Best fine-tuned run: `{best['name']}` at {pct(float(best['exact_match_accuracy']))}.",
                "",
            ]
        )
        failures = best.get("failed_examples_preview", [])
        if failures:
            lines.extend(["## Best Run Failure Preview", ""])
            for failure in failures:
                prediction = str(failure.get("prediction", "")).splitlines()[0]
                lines.extend(
                    [
                        f"- `{failure.get('id')}`",
                        f"  - Gold: `{failure.get('gold')}`",
                        f"  - Prediction: {prediction}",
                    ]
                )
            lines.append("")
    lines.extend(
        [
            "## Limitations",
            "",
            "This is a narrow synthetic benchmark for 2x2 pure-strategy Nash equilibria only. "
            "It does not demonstrate broad game-theory reasoning improvement.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GT-Bench scorer reports.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_named_report, default=[])
    parser.add_argument("--confirmation-gold", type=Path, default=None)
    parser.add_argument("--confirmation-baseline", type=Path, default=None)
    parser.add_argument("--confirmation-run", action="append", type=parse_named_report, default=[])
    parser.add_argument("--out-json", type=Path, default=Path("reports/gt_bench_results.json"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/gt_bench_results.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(load_config(args.config), args.baseline, args.run)
    add_confirmation_summary(
        summary,
        args.confirmation_baseline,
        args.confirmation_run,
        args.confirmation_gold,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, summary)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
