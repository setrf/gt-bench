from __future__ import annotations

import argparse
import json
import math
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


def ci_text(row: dict[str, Any]) -> str:
    ci = row["accuracy_ci_95"]
    return f"{pct(float(ci['lower']))}-{pct(float(ci['upper']))}"


def binomial_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def binomial_sf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def exact_binomial_ci(k: int, n: int, alpha: float = 0.05) -> dict[str, float | str]:
    if n <= 0:
        raise ValueError("n must be positive")

    target = alpha / 2
    lower = 0.0
    upper = 1.0

    if k > 0:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if binomial_sf(k, n, mid) < target:
                lo = mid
            else:
                hi = mid
        lower = hi

    if k < n:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if binomial_cdf(k, n, mid) > target:
                lo = mid
            else:
                hi = mid
        upper = hi

    return {
        "method": "clopper_pearson",
        "confidence": 1 - alpha,
        "lower": lower,
        "upper": upper,
    }


def bucket_deltas(
    buckets: dict[str, dict[str, float | int]],
    baseline_buckets: dict[str, dict[str, float | int]],
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for count in sorted(set(buckets) | set(baseline_buckets), key=int):
        if count not in buckets or count not in baseline_buckets:
            continue
        deltas[count] = float(buckets[count]["accuracy"]) - float(baseline_buckets[count]["accuracy"])
    return deltas


def summarize_report(
    report: dict[str, Any],
    base_acc: float | None = None,
    base_buckets: dict[str, dict[str, float | int]] | None = None,
) -> dict[str, Any]:
    total = int(report["num_correct"]) + int(report["num_incorrect"])
    acc = float(report["exact_match_accuracy"])
    summary = {
        "exact_match_accuracy": acc,
        "accuracy_ci_95": exact_binomial_ci(int(report["num_correct"]), total),
        "num_correct": report["num_correct"],
        "num_incorrect": report["num_incorrect"],
        "total_examples": total,
        "accuracy_by_number_of_equilibria": report["accuracy_by_number_of_equilibria"],
        "failed_examples_preview": report.get("failed_examples", [])[:10],
    }
    if base_acc is not None:
        summary["delta_vs_baseline"] = acc - base_acc
        summary["delta_pp_vs_baseline"] = (acc - base_acc) * 100
    if base_buckets is not None:
        summary["equilibrium_count_delta_vs_baseline"] = bucket_deltas(
            report["accuracy_by_number_of_equilibria"],
            base_buckets,
        )
    return summary


def build_summary(config: dict[str, Any], baseline_path: Path, runs: list[tuple[str, Path]]) -> dict[str, Any]:
    baseline = read_report(baseline_path)
    run_summaries = []
    base_acc = float(baseline["exact_match_accuracy"])
    base_buckets = baseline["accuracy_by_number_of_equilibria"]

    for name, path in runs:
        report = read_report(path)
        run_summary = summarize_report(report, base_acc, base_buckets)
        run_summary["name"] = name
        run_summary["report_path"] = str(path)
        run_summaries.append(run_summary)

    best = max(run_summaries, key=lambda row: row["exact_match_accuracy"], default=None)
    test_path = Path(config["data"]["test"])
    return {
        "model_id": config["model_id"],
        "baseline_report": str(baseline_path),
        "test_path": str(test_path),
        "test_sha256": sha256_file(test_path) if test_path.exists() else None,
        "baseline": summarize_report(baseline, base_acc, base_buckets),
        "runs": run_summaries,
        "best_run": best,
    }


def add_confirmation_summary(
    summary: dict[str, Any],
    baseline_path: Path | None,
    runs: list[tuple[str, Path]],
    gold_path: Path | None,
) -> None:
    add_extra_summary(summary, "confirmation", baseline_path, runs, gold_path)


def add_extra_summary(
    summary: dict[str, Any],
    key: str,
    baseline_path: Path | None,
    runs: list[tuple[str, Path]],
    gold_path: Path | None,
) -> None:
    if baseline_path is None:
        return

    baseline = read_report(baseline_path)
    base_acc = float(baseline["exact_match_accuracy"])
    base_buckets = baseline["accuracy_by_number_of_equilibria"]
    run_summaries = []
    for name, path in runs:
        report = read_report(path)
        run_summary = summarize_report(report, base_acc, base_buckets)
        run_summary["name"] = name
        run_summary["report_path"] = str(path)
        run_summaries.append(run_summary)

    summary[key] = {
        "gold_path": str(gold_path) if gold_path else None,
        "gold_sha256": sha256_file(gold_path) if gold_path and gold_path.exists() else None,
        "baseline_report": str(baseline_path),
        "baseline": summarize_report(baseline, base_acc, base_buckets),
        "runs": run_summaries,
        "best_run": max(run_summaries, key=lambda row: row["exact_match_accuracy"], default=None),
    }


def append_accuracy_table(lines: list[str], baseline: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "| Run | Accuracy | 95% exact binomial CI | Correct | Incorrect | Delta vs baseline |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.append(
        f"| baseline | {pct(float(baseline['exact_match_accuracy']))} | "
        f"{ci_text(baseline)} | {baseline['num_correct']} | {baseline['num_incorrect']} | 0.00 pp |"
    )
    for run in runs:
        delta_pp = float(run["delta_vs_baseline"]) * 100
        lines.append(
            f"| {run['name']} | {pct(float(run['exact_match_accuracy']))} | "
            f"{ci_text(run)} | {run['num_correct']} | {run['num_incorrect']} | {delta_pp:+.2f} pp |"
        )


def append_bucket_tables(
    lines: list[str],
    heading: str,
    baseline: dict[str, Any],
    runs: list[dict[str, Any]],
) -> None:
    lines.extend(["", f"## {heading}", ""])
    for label, report in [("baseline", baseline), *[(run["name"], run) for run in runs]]:
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
    append_bucket_tables(lines, "Accuracy By Number Of Equilibria", baseline, summary["runs"])

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

    stress = summary.get("stress")
    if stress:
        lines.extend(["## Balanced Stress Evaluation", ""])
        if stress.get("gold_path"):
            lines.append(f"Stress set: `{stress['gold_path']}`")
        if stress.get("gold_sha256"):
            lines.append(f"Stress SHA-256: `{stress['gold_sha256']}`")
        lines.append("")
        lines.append(
            "The stress set contains equal numbers of examples with 0, 1, 2, 3, and 4 "
            "pure-strategy equilibria."
        )
        lines.append("")
        append_accuracy_table(lines, stress["baseline"], stress["runs"])
        append_bucket_tables(
            lines,
            "Stress Accuracy By Number Of Equilibria",
            stress["baseline"],
            stress["runs"],
        )

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
    parser.add_argument("--stress-gold", type=Path, default=None)
    parser.add_argument("--stress-baseline", type=Path, default=None)
    parser.add_argument("--stress-run", action="append", type=parse_named_report, default=[])
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
    add_extra_summary(summary, "stress", args.stress_baseline, args.stress_run, args.stress_gold)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, summary)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
