from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

from summarize_results import exact_binomial_ci, pct
from tinker_common import CONFIG_PATH, load_config, sha256_file


DEFAULT_SEEDS = (42, 1009, 2027)
DEFAULT_SIZES = (250, 1000, 5000)
PUBLIC_PATH = Path("reports/seed_sweep_results.json")
PUBLIC_RESULTS_PATH = Path("reports/gt_bench_results.json")


def report_name(seed: int, size: int) -> str:
    if seed == 42:
        return f"qwen36_27b_sft_{size:04d}"
    return f"qwen36_27b_seed{seed}_sft_{size:04d}"


def default_report_path(seed: int, size: int) -> Path:
    return Path("reports") / f"{report_name(seed, size)}_report.json"


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_report(path: Path, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    report = read_json(path) or fallback
    if report is None:
        return {"status": "pending", "report_path": str(path)}
    if report.get("status") == "pending":
        return {"status": "pending", "report_path": str(path)}

    total = int(report.get("total_examples", int(report["num_correct"]) + int(report["num_incorrect"])))
    correct = int(report["num_correct"])
    return {
        "status": "complete",
        "report_path": str(path) if path.exists() else report.get("report_path", str(path)),
        "total_examples": total,
        "exact_match_accuracy": float(report["exact_match_accuracy"]),
        "accuracy_ci_95": report.get("accuracy_ci_95", exact_binomial_ci(correct, total)),
        "num_correct": correct,
        "num_incorrect": int(report["num_incorrect"]),
        "accuracy_by_number_of_equilibria": report.get("accuracy_by_number_of_equilibria", {}),
        "failed_examples_preview": report.get("failed_examples_preview", report.get("failed_examples", [])[:10]),
    }


def public_fallbacks(path: Path = PUBLIC_PATH) -> dict[tuple[int, int], dict[str, Any]]:
    summary = read_json(path)
    if not summary:
        return {}
    runs = summary.get("runs", {})
    if not isinstance(runs, dict):
        return {}

    fallbacks: dict[tuple[int, int], dict[str, Any]] = {}
    for size_text, rows in runs.items():
        if not isinstance(rows, list):
            continue
        try:
            size = int(size_text)
        except ValueError:
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("status") == "complete":
                fallbacks[(int(row["seed"]), size)] = row
    return fallbacks


def public_baseline_fallback(path: Path = PUBLIC_RESULTS_PATH) -> dict[str, Any] | None:
    summary = read_json(path)
    if not summary:
        return None
    baseline = summary.get("baseline")
    return baseline if isinstance(baseline, dict) else None


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def t_critical_95(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }
    if df <= 0:
        return 0.0
    return table.get(df, 1.96)


def clamp(value: float, lower: float | None, upper: float | None) -> float:
    if lower is not None:
        value = max(lower, value)
    if upper is not None:
        value = min(upper, value)
    return value


def t_interval(
    values: Sequence[float],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> dict[str, float | str]:
    m = mean(values)
    if len(values) < 2:
        return {"method": "t_interval_seed_mean", "confidence": 0.95, "lower": m, "upper": m}
    half_width = t_critical_95(len(values) - 1) * sample_std(values) / math.sqrt(len(values))
    return {
        "method": "t_interval_seed_mean",
        "confidence": 0.95,
        "lower": clamp(m - half_width, lower_bound, upper_bound),
        "upper": clamp(m + half_width, lower_bound, upper_bound),
    }


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    iterations: int = 10_000,
    seed: int = 1729,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> dict[str, float | int | str]:
    if not values:
        raise ValueError("cannot bootstrap an empty sequence")
    if len(values) == 1:
        return {
            "method": "paired_seed_bootstrap",
            "confidence": 0.95,
            "iterations": iterations,
            "lower": clamp(values[0], lower_bound, upper_bound),
            "upper": clamp(values[0], lower_bound, upper_bound),
        }
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(sample))
    means.sort()
    lower_index = int(0.025 * iterations)
    upper_index = min(iterations - 1, int(0.975 * iterations))
    return {
        "method": "paired_seed_bootstrap",
        "confidence": 0.95,
        "iterations": iterations,
        "lower": clamp(means[lower_index], lower_bound, upper_bound),
        "upper": clamp(means[upper_index], lower_bound, upper_bound),
    }


def summarize_values(
    values: Sequence[float],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_std": sample_std(values),
        "min": min(values),
        "max": max(values),
        "t_interval_95": t_interval(values, lower_bound, upper_bound),
        "bootstrap_interval_95": bootstrap_mean_interval(
            values,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        ),
    }


def build_statistics(runs: dict[int, list[dict[str, Any]]], baseline_accuracy: float) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    by_seed_and_size = {
        (int(row["seed"]), size): float(row["exact_match_accuracy"])
        for size, rows in runs.items()
        for row in rows
        if row.get("status") == "complete"
    }

    for size, rows in runs.items():
        accuracies = [float(row["exact_match_accuracy"]) for row in rows if row.get("status") == "complete"]
        if not accuracies:
            stats[str(size)] = {"status": "pending", "n": 0}
            continue
        deltas = [value - baseline_accuracy for value in accuracies]
        stats[str(size)] = {
            "status": "complete",
            "accuracy": summarize_values(accuracies, 0.0, 1.0),
            "delta_vs_baseline": summarize_values(deltas),
            "mean_delta_pp_vs_baseline": mean(deltas) * 100,
        }

    comparisons: dict[str, Any] = {}
    for high, low in [(1000, 250), (5000, 1000), (5000, 250)]:
        common_seeds = sorted(
            seed
            for seed in {seed for seed, size in by_seed_and_size if size == high}
            if (seed, low) in by_seed_and_size
        )
        if not common_seeds:
            comparisons[f"{high}_vs_{low}"] = {"status": "pending", "n": 0}
            continue
        deltas = [by_seed_and_size[(seed, high)] - by_seed_and_size[(seed, low)] for seed in common_seeds]
        comparisons[f"{high}_vs_{low}"] = {
            "status": "complete",
            "n": len(deltas),
            "seeds": common_seeds,
            "mean_delta": mean(deltas),
            "mean_delta_pp": mean(deltas) * 100,
            "sample_std_delta": sample_std(deltas),
            "t_interval_95_delta": t_interval(deltas),
            "bootstrap_interval_95_delta": bootstrap_mean_interval(deltas),
        }
    stats["paired_comparisons"] = comparisons
    return stats


def build_summary(
    config: dict[str, Any],
    baseline_path: Path,
    seeds: Sequence[int],
    sizes: Sequence[int],
    report_paths: dict[tuple[int, int], Path] | None = None,
    fallback_path: Path = PUBLIC_PATH,
    public_results_path: Path = PUBLIC_RESULTS_PATH,
) -> dict[str, Any]:
    baseline_report = read_json(baseline_path) or public_baseline_fallback(public_results_path)
    if baseline_report is None:
        raise FileNotFoundError(f"baseline report not found: {baseline_path}")
    baseline = summarize_report(baseline_path, baseline_report)
    baseline_accuracy = float(baseline["exact_match_accuracy"])
    fallbacks = public_fallbacks(fallback_path)

    runs: dict[int, list[dict[str, Any]]] = {}
    report_paths = report_paths or {}
    for size in sizes:
        rows = []
        for seed in seeds:
            path = report_paths.get((seed, size), default_report_path(seed, size))
            row = summarize_report(path, fallbacks.get((seed, size)))
            row["seed"] = seed
            row["train_size"] = size
            row["name"] = report_name(seed, size)
            if row["status"] == "complete":
                row["delta_vs_baseline"] = float(row["exact_match_accuracy"]) - baseline_accuracy
                row["delta_pp_vs_baseline"] = row["delta_vs_baseline"] * 100
            rows.append(row)
        runs[size] = rows

    complete = all(row["status"] == "complete" for rows in runs.values() for row in rows)
    test_path = Path(config["data"]["test"])
    public_summary = read_json(fallback_path) or {}
    test_sha = sha256_file(test_path) if test_path.exists() else public_summary.get("test_sha256")
    return {
        "model_id": config["model_id"],
        "status": "complete" if complete else "pending",
        "test_path": str(test_path),
        "test_sha256": test_sha,
        "baseline_report": str(baseline_path),
        "baseline": baseline,
        "seeds": list(seeds),
        "train_sizes": list(sizes),
        "runs": {str(size): runs[size] for size in sizes},
        "statistics": build_statistics(runs, baseline_accuracy),
    }


def stable_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, list):
        return [stable_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: stable_floats(item) for key, item in value.items()}
    return value


def interval_text(interval: dict[str, Any], scale: float = 1.0) -> str:
    return f"{float(interval['lower']) * scale:.2f}-{float(interval['upper']) * scale:.2f}"


def signed_interval_text(interval: dict[str, Any], scale: float = 1.0) -> str:
    return f"{float(interval['lower']) * scale:+.2f} to {float(interval['upper']) * scale:+.2f}"


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GT-Bench Repeated-Seed Learning Curve",
        "",
        f"Status: `{summary['status']}`",
        f"Model: `{summary['model_id']}`",
        f"Fixed test set: `{summary['test_path']}`",
        "",
        "This summary repeats the SFT training-size sweep across independent training-data seeds while keeping the canonical 500-example test set fixed. Seed `42` is the original public sweep; the additional seeds test whether the learning curve is stable under regenerated synthetic training data.",
        "",
        "## Per-Seed Accuracy",
        "",
        "| Train size | Seed | Accuracy | Correct | Incorrect | Delta vs baseline |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline_acc = float(summary["baseline"]["exact_match_accuracy"])
    for size in summary["train_sizes"]:
        for row in summary["runs"][str(size)]:
            if row["status"] != "complete":
                lines.append(f"| {size} | {row['seed']} | pending |  |  |  |")
                continue
            delta = float(row["exact_match_accuracy"]) - baseline_acc
            lines.append(
                f"| {size} | {row['seed']} | {pct(float(row['exact_match_accuracy']))} | "
                f"{row['num_correct']} | {row['num_incorrect']} | {delta * 100:+.2f} pp |"
            )

    lines.extend(
        [
            "",
            "## Aggregate Statistics",
            "",
            "| Train size | n | Mean accuracy | Seed SD | 95% t CI for mean accuracy | Mean delta vs baseline | 95% bootstrap CI for mean delta |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for size in summary["train_sizes"]:
        stats = summary["statistics"][str(size)]
        if stats["status"] != "complete":
            lines.append(f"| {size} | 0 | pending | pending | pending | pending | pending |")
            continue
        acc = stats["accuracy"]
        delta = stats["delta_vs_baseline"]
        lines.append(
            f"| {size} | {acc['n']} | {pct(float(acc['mean']))} | "
            f"{float(acc['sample_std']) * 100:.2f} pp | "
            f"{interval_text(acc['t_interval_95'], 100)}% | "
            f"{float(stats['mean_delta_pp_vs_baseline']):+.2f} pp | "
            f"{signed_interval_text(delta['bootstrap_interval_95'], 100)} pp |"
        )

    lines.extend(["", "## Paired Training-Size Comparisons", ""])
    lines.extend(
        [
            "| Comparison | n | Mean delta | 95% t CI | 95% bootstrap CI |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in summary["statistics"]["paired_comparisons"].items():
        if row["status"] != "complete":
            lines.append(f"| {name.replace('_', ' ')} | 0 | pending | pending | pending |")
            continue
        lines.append(
            f"| {name.replace('_', ' ')} | {row['n']} | {float(row['mean_delta_pp']):+.2f} pp | "
            f"{signed_interval_text(row['t_interval_95_delta'], 100)} pp | "
            f"{signed_interval_text(row['bootstrap_interval_95_delta'], 100)} pp |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a seed-level comparison, not a new task. It supports a stronger learning-curve claim when the 1000- and 5000-example means remain above baseline across independently generated training sets. With only three seeds, uncertainty intervals should be read as descriptive rather than definitive.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GT-Bench repeated-seed SFT reports.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--baseline", type=Path, default=Path("reports/baseline_qwen36_27b_report.json"))
    parser.add_argument("--seed", dest="seeds", type=int, action="append", default=[])
    parser.add_argument("--size", dest="sizes", type=int, action="append", default=[])
    parser.add_argument("--out-json", type=Path, default=PUBLIC_PATH)
    parser.add_argument("--out-md", type=Path, default=Path("reports/seed_sweep_results.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = args.seeds or list(DEFAULT_SEEDS)
    sizes = args.sizes or list(DEFAULT_SIZES)
    summary = build_summary(load_config(args.config), args.baseline, seeds, sizes, fallback_path=args.out_json)
    summary = stable_floats(summary)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, summary)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
