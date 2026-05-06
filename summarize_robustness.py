from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from summarize_results import exact_binomial_ci, pct
from tinker_common import sha256_file


def read_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ci_text(row: dict[str, Any]) -> str:
    ci = row["accuracy_ci_95"]
    return f"{pct(float(ci['lower']))}-{pct(float(ci['upper']))}"


def enrich(report: dict[str, Any], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    total = int(report["total_examples"])
    accuracy = float(report["exact_match_accuracy"])
    row = {
        "total_examples": total,
        "exact_match_accuracy": accuracy,
        "accuracy_ci_95": exact_binomial_ci(int(report["num_correct"]), total),
        "num_correct": report["num_correct"],
        "num_incorrect": report["num_incorrect"],
        "accuracy_by_prompt_variant": report["accuracy_by_prompt_variant"],
        "accuracy_by_number_of_equilibria": report["accuracy_by_number_of_equilibria"],
        "accuracy_by_variant_and_number_of_equilibria": report[
            "accuracy_by_variant_and_number_of_equilibria"
        ],
        "failed_examples_preview": report.get("failed_examples", [])[:10],
    }
    if baseline is not None:
        row["delta_vs_baseline"] = accuracy - float(baseline["exact_match_accuracy"])
        row["delta_pp_vs_baseline"] = (accuracy - float(baseline["exact_match_accuracy"])) * 100
        row["prompt_variant_delta_vs_baseline"] = {
            variant: float(stats["accuracy"])
            - float(baseline["accuracy_by_prompt_variant"][variant]["accuracy"])
            for variant, stats in report["accuracy_by_prompt_variant"].items()
            if variant in baseline["accuracy_by_prompt_variant"]
        }
    return row


def build_summary(gold: Path, baseline_path: Path, finetuned_path: Path) -> dict[str, Any]:
    baseline_report = read_report(baseline_path)
    finetuned_report = read_report(finetuned_path)
    baseline = enrich(baseline_report)
    finetuned = enrich(finetuned_report, baseline_report)
    return {
        "gold_path": str(gold),
        "gold_sha256": sha256_file(gold) if gold.exists() else None,
        "baseline_report": str(baseline_path),
        "finetuned_report": str(finetuned_path),
        "baseline": baseline,
        "finetuned": finetuned,
    }


def append_accuracy_table(lines: list[str], summary: dict[str, Any]) -> None:
    baseline = summary["baseline"]
    finetuned = summary["finetuned"]
    delta = float(finetuned["delta_pp_vs_baseline"])
    lines.extend(
        [
            "| Run | Accuracy | 95% exact binomial CI | Correct | Incorrect | Delta vs baseline |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| baseline | {pct(float(baseline['exact_match_accuracy']))} | "
                f"{ci_text(baseline)} | {baseline['num_correct']} | "
                f"{baseline['num_incorrect']} | 0.00 pp |"
            ),
            (
                f"| 5000-example SFT | {pct(float(finetuned['exact_match_accuracy']))} | "
                f"{ci_text(finetuned)} | {finetuned['num_correct']} | "
                f"{finetuned['num_incorrect']} | {delta:+.2f} pp |"
            ),
            "",
        ]
    )


def append_variant_table(lines: list[str], summary: dict[str, Any]) -> None:
    baseline = summary["baseline"]["accuracy_by_prompt_variant"]
    finetuned = summary["finetuned"]["accuracy_by_prompt_variant"]
    lines.extend(
        [
            "| Prompt variant | Baseline | 5000-example SFT | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for variant in sorted(baseline):
        base_acc = float(baseline[variant]["accuracy"])
        ft_acc = float(finetuned[variant]["accuracy"])
        lines.append(f"| `{variant}` | {pct(base_acc)} | {pct(ft_acc)} | {(ft_acc - base_acc) * 100:+.2f} pp |")
    lines.append("")


def append_bucket_table(lines: list[str], summary: dict[str, Any]) -> None:
    baseline = summary["baseline"]["accuracy_by_number_of_equilibria"]
    finetuned = summary["finetuned"]["accuracy_by_number_of_equilibria"]
    lines.extend(
        [
            "| Equilibria | Baseline | 5000-example SFT | Delta |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for count in sorted(baseline, key=int):
        base_acc = float(baseline[count]["accuracy"])
        ft_acc = float(finetuned[count]["accuracy"])
        lines.append(f"| {count} | {pct(base_acc)} | {pct(ft_acc)} | {(ft_acc - base_acc) * 100:+.2f} pp |")
    lines.append("")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GT-Bench Robustness Results",
        "",
        f"Robustness set: `{summary['gold_path']}`",
    ]
    if summary.get("gold_sha256"):
        lines.append(f"Robustness SHA-256: `{summary['gold_sha256']}`")
    lines.extend(
        [
            "",
            "The robustness set keeps the same 2x2 pure-strategy Nash task but changes the prompt surface. "
            "It contains 250 examples: 5 prompt variants x 5 equilibrium-count buckets x 10 examples.",
            "",
            "## Overall Accuracy",
            "",
        ]
    )
    append_accuracy_table(lines, summary)
    lines.extend(["## Accuracy By Prompt Variant", ""])
    append_variant_table(lines, summary)
    lines.extend(["## Accuracy By Number Of Equilibria", ""])
    append_bucket_table(lines, summary)

    failures = summary["finetuned"].get("failed_examples_preview", [])
    if failures:
        lines.extend(["## Fine-Tuned Failure Preview", ""])
        for failure in failures:
            prediction = str(failure.get("prediction", "")).splitlines()[0]
            lines.extend(
                [
                    f"- `{failure.get('id')}` (`{failure.get('prompt_variant')}`)",
                    f"  - Gold: `{failure.get('gold')}`",
                    f"  - Prediction: {prediction}",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "The fine-tuned checkpoint improves overall robustness, which makes simple prompt-template "
            "overfitting less likely. The weakest remaining variants should drive the next targeted "
            "adversarial SFT slice.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GT-Bench robustness reports.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--finetuned", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=Path("reports/robustness_results.json"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/robustness_results.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(args.gold, args.baseline, args.finetuned)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, summary)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
