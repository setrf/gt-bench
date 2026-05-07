from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from summarize_results import exact_binomial_ci, pct


EVALUATIONS = (
    "canonical",
    "confirmation",
    "stress",
    "robustness",
)
PUBLIC_RESULTS_PATH = Path("reports/gt_bench_results.json")
PUBLIC_ROBUSTNESS_PATH = Path("reports/robustness_results.json")


def read_report(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ci_text(row: dict[str, Any]) -> str:
    ci = row["accuracy_ci_95"]
    return f"{pct(float(ci['lower']))}-{pct(float(ci['upper']))}"


def summarize_report(path: Path | None, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    report = read_report(path) or fallback
    if report is None:
        return {"status": "pending", "report_path": str(path) if path else None}

    total = int(report.get("total_examples", int(report["num_correct"]) + int(report["num_incorrect"])))
    correct = int(report["num_correct"])
    summary: dict[str, Any] = {
        "status": "complete",
        "report_path": str(path) if path and path.exists() else report.get("report_path"),
        "total_examples": total,
        "exact_match_accuracy": float(report["exact_match_accuracy"]),
        "accuracy_ci_95": report.get("accuracy_ci_95", exact_binomial_ci(correct, total)),
        "num_correct": correct,
        "num_incorrect": int(report["num_incorrect"]),
        "accuracy_by_number_of_equilibria": report.get("accuracy_by_number_of_equilibria", {}),
        "failed_examples_preview": report.get(
            "failed_examples_preview",
            report.get("failed_examples", [])[:10],
        ),
    }
    if "accuracy_by_prompt_variant" in report:
        summary["accuracy_by_prompt_variant"] = report["accuracy_by_prompt_variant"]
    if "accuracy_by_variant_and_number_of_equilibria" in report:
        summary["accuracy_by_variant_and_number_of_equilibria"] = report[
            "accuracy_by_variant_and_number_of_equilibria"
        ]
    return summary


def add_delta(row: dict[str, Any]) -> None:
    original = row["original_5000_sft"]
    adversarial = row["adversarial_sft"]
    if original["status"] != "complete" or adversarial["status"] != "complete":
        row["delta_pp_vs_original_5000_sft"] = None
        return
    delta = float(adversarial["exact_match_accuracy"]) - float(original["exact_match_accuracy"])
    row["delta_vs_original_5000_sft"] = delta
    row["delta_pp_vs_original_5000_sft"] = delta * 100


def variant_deltas(row: dict[str, Any]) -> dict[str, float] | None:
    original = row["original_5000_sft"]
    adversarial = row["adversarial_sft"]
    if original["status"] != "complete" or adversarial["status"] != "complete":
        return None
    original_variants = original.get("accuracy_by_prompt_variant")
    adversarial_variants = adversarial.get("accuracy_by_prompt_variant")
    if not isinstance(original_variants, dict) or not isinstance(adversarial_variants, dict):
        return None
    return {
        variant: (
            float(adversarial_variants[variant]["accuracy"])
            - float(original_variants[variant]["accuracy"])
        )
        * 100
        for variant in sorted(original_variants)
        if variant in adversarial_variants
    }


def public_original_fallbacks(
    results_path: Path = PUBLIC_RESULTS_PATH,
    robustness_path: Path = PUBLIC_ROBUSTNESS_PATH,
) -> dict[str, dict[str, Any]]:
    fallbacks: dict[str, dict[str, Any]] = {}
    results = read_report(results_path)
    if results:
        if results.get("best_run"):
            fallbacks["canonical"] = results["best_run"]
        if results.get("confirmation", {}).get("best_run"):
            fallbacks["confirmation"] = results["confirmation"]["best_run"]
        if results.get("stress", {}).get("best_run"):
            fallbacks["stress"] = results["stress"]["best_run"]
    robustness = read_report(robustness_path)
    if robustness and robustness.get("finetuned"):
        row = dict(robustness["finetuned"])
        row.setdefault("report_path", robustness.get("finetuned_report"))
        fallbacks["robustness"] = row
    return fallbacks


def build_summary(
    paths: dict[str, tuple[Path | None, Path | None]],
    original_fallbacks: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    original_fallbacks = original_fallbacks or {}
    evaluations: dict[str, Any] = {}
    for name in EVALUATIONS:
        original_path, adversarial_path = paths[name]
        row = {
            "original_5000_sft": summarize_report(original_path, original_fallbacks.get(name)),
            "adversarial_sft": summarize_report(adversarial_path),
        }
        add_delta(row)
        if name == "robustness":
            row["prompt_variant_delta_pp_vs_original"] = variant_deltas(row)
        evaluations[name] = row

    robust_adv = evaluations["robustness"]["adversarial_sft"]
    canonical_original = evaluations["canonical"]["original_5000_sft"]
    canonical_adv = evaluations["canonical"]["adversarial_sft"]
    acceptance = {
        "robustness_target_accuracy": 0.95,
        "max_allowed_canonical_regression_pp": -0.5,
        "robustness_target_met": None,
        "canonical_regression_within_bound": None,
    }
    if robust_adv["status"] == "complete":
        acceptance["robustness_target_met"] = float(robust_adv["exact_match_accuracy"]) >= 0.95
    if canonical_original["status"] == "complete" and canonical_adv["status"] == "complete":
        delta_pp = (
            float(canonical_adv["exact_match_accuracy"])
            - float(canonical_original["exact_match_accuracy"])
        ) * 100
        acceptance["canonical_regression_pp"] = delta_pp
        acceptance["canonical_regression_within_bound"] = delta_pp >= -0.5

    return {
        "run_name": "qwen36_27b_sft_5000_plus_prompt_adv",
        "model_id": "Qwen/Qwen3.6-27B",
        "status": (
            "complete"
            if all(
                evaluations[name]["adversarial_sft"]["status"] == "complete"
                for name in EVALUATIONS
            )
            else "pending"
        ),
        "evaluations": evaluations,
        "acceptance": acceptance,
    }


def append_eval_table(lines: list[str], summary: dict[str, Any]) -> None:
    lines.extend(
        [
            "| Evaluation | Original 5000 SFT | Adversarial SFT | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name in EVALUATIONS:
        row = summary["evaluations"][name]
        original = row["original_5000_sft"]
        adversarial = row["adversarial_sft"]
        original_text = (
            pct(float(original["exact_match_accuracy"]))
            if original["status"] == "complete"
            else "pending"
        )
        adversarial_text = (
            pct(float(adversarial["exact_match_accuracy"]))
            if adversarial["status"] == "complete"
            else "pending"
        )
        delta = row.get("delta_pp_vs_original_5000_sft")
        delta_text = f"{float(delta):+.2f} pp" if delta is not None else "pending"
        lines.append(f"| {name} | {original_text} | {adversarial_text} | {delta_text} |")
    lines.append("")


def append_robustness_variant_table(lines: list[str], summary: dict[str, Any]) -> None:
    robust = summary["evaluations"]["robustness"]
    original = robust["original_5000_sft"]
    adversarial = robust["adversarial_sft"]
    original_variants = original.get("accuracy_by_prompt_variant")
    if not isinstance(original_variants, dict):
        return

    adversarial_variants = adversarial.get("accuracy_by_prompt_variant", {})
    lines.extend(
        [
            "## Robustness By Prompt Variant",
            "",
            "| Prompt variant | Original 5000 SFT | Adversarial SFT | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for variant in sorted(original_variants):
        old_acc = float(original_variants[variant]["accuracy"])
        if isinstance(adversarial_variants, dict) and variant in adversarial_variants:
            adv_acc = float(adversarial_variants[variant]["accuracy"])
            adv_text = pct(adv_acc)
            delta_text = f"{(adv_acc - old_acc) * 100:+.2f} pp"
        else:
            adv_text = "pending"
            delta_text = "pending"
        lines.append(f"| `{variant}` | {pct(old_acc)} | {adv_text} | {delta_text} |")
    lines.append("")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    status = str(summary["status"])
    lines = [
        "# GT-Bench Adversarial SFT Follow-Up",
        "",
        f"Status: `{status}`",
        "",
        "This follow-up keeps the mathematical task unchanged and targets the prompt-format weakness exposed by the robustness set. The adversarial training supplement emphasizes compact payoff pairs, JSON-like payoff objects, answer-only prompts, minimal matrices, and balanced equilibrium-count buckets.",
        "",
    ]
    if status == "pending":
        lines.extend(
            [
                "The public pipeline is implemented, but the new Tinker SFT/evaluation run has not been completed in this checkout yet.",
                "",
            ]
        )
    lines.extend(["## Accuracy Summary", ""])
    append_eval_table(lines, summary)
    append_robustness_variant_table(lines, summary)
    lines.extend(
        [
            "## Acceptance Criteria",
            "",
            "- Target robustness accuracy: at least 95.00%.",
            "- Canonical regression tolerance: no worse than -0.50 percentage points versus the original 5000-example SFT checkpoint.",
            "- Main expected gains: `compact_pairs` and `json_payoffs`.",
            "",
            "## Claim Boundaries",
            "",
            "This follow-up can support a targeted robustness claim for one synthetic formal task. It does not claim broad game-theory reasoning, and it still excludes mixed strategies, dominance, welfare, sequential games, and natural-language story problems.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GT-Bench adversarial SFT follow-up reports.")
    parser.add_argument("--canonical-original", type=Path, default=Path("reports/qwen36_27b_sft_5000_report.json"))
    parser.add_argument("--canonical-adversarial", type=Path, default=Path("reports/qwen36_27b_sft_5000_plus_prompt_adv_report.json"))
    parser.add_argument("--confirmation-original", type=Path, default=Path("reports/confirm_qwen36_27b_sft_5000_seed20260505_report.json"))
    parser.add_argument("--confirmation-adversarial", type=Path, default=Path("reports/confirm_qwen36_27b_sft_5000_plus_prompt_adv_seed20260505_report.json"))
    parser.add_argument("--stress-original", type=Path, default=Path("reports/stress_qwen36_27b_sft_5000_seed314159_report.json"))
    parser.add_argument("--stress-adversarial", type=Path, default=Path("reports/stress_qwen36_27b_sft_5000_plus_prompt_adv_seed314159_report.json"))
    parser.add_argument("--robustness-original", type=Path, default=Path("reports/robust_qwen36_27b_sft_5000_seed271828_report.json"))
    parser.add_argument("--robustness-adversarial", type=Path, default=Path("reports/robust_qwen36_27b_sft_5000_plus_prompt_adv_seed271828_report.json"))
    parser.add_argument("--public-results", type=Path, default=PUBLIC_RESULTS_PATH)
    parser.add_argument("--public-robustness", type=Path, default=PUBLIC_ROBUSTNESS_PATH)
    parser.add_argument("--out-json", type=Path, default=Path("reports/adversarial_results.json"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/adversarial_results.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(
        {
            "canonical": (args.canonical_original, args.canonical_adversarial),
            "confirmation": (args.confirmation_original, args.confirmation_adversarial),
            "stress": (args.stress_original, args.stress_adversarial),
            "robustness": (args.robustness_original, args.robustness_adversarial),
        },
        public_original_fallbacks(args.public_results, args.public_robustness),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, summary)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
