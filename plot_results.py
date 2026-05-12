from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Sequence


SUMMARY_PATH = Path("reports/gt_bench_results.json")
ROBUSTNESS_PATH = Path("reports/robustness_results.json")
ADVERSARIAL_PATH = Path("reports/adversarial_results.json")
SEED_SWEEP_PATH = Path("reports/seed_sweep_results.json")
FIGURE_DIR = Path("reports/figures")

BLUE = "#2563eb"
TEAL = "#0f766e"
ORANGE = "#ea580c"
RED = "#dc2626"
GRAY = "#64748b"
INK = "#111827"
MUTED = "#475569"
GRID = "#dbe3ef"
PAPER = "#ffffff"


def require(row: dict[str, Any], key: str, context: str) -> Any:
    if key not in row:
        raise KeyError(f"missing required key {key!r} in {context}")
    return row[key]


def accuracy(row: dict[str, Any], context: str) -> float:
    value = require(row, "exact_match_accuracy", context)
    if not isinstance(value, int | float):
        raise TypeError(f"{context}.exact_match_accuracy must be numeric")
    return float(value)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(
    x: float,
    y: float,
    value: object,
    size: int = 14,
    fill: str = INK,
    anchor: str = "start",
    weight: str = "400",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, Helvetica, sans-serif" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}">{esc(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str = GRID, width: float = 1) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width:.1f}" />'
    )


def rect(x: float, y: float, width: float, height: float, fill: str, radius: float = 2) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{radius:.1f}" fill="{fill}" />'
    )


def svg_frame(width: int, height: int, title: str, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
            f"<title>{esc(title)}</title>",
            rect(0, 0, width, height, PAPER, 0),
            *body,
            "</svg>",
            "",
        ]
    )


def y_scale(value: float, top: float, bottom: float) -> float:
    return bottom - max(0.0, min(1.0, value)) * (bottom - top)


def add_axes(body: list[str], left: float, top: float, right: float, bottom: float) -> None:
    for tick in range(0, 101, 20):
        y = y_scale(tick / 100, top, bottom)
        body.append(line(left, y, right, y))
        body.append(text(left - 12, y + 5, f"{tick}%", size=14, fill=MUTED, anchor="end"))
    body.append(line(left, top, left, bottom, color=INK, width=1.2))
    body.append(line(left, bottom, right, bottom, color=INK, width=1.2))


def human_run_name(name: str) -> str:
    if "0250" in name:
        return "250 SFT"
    if "1000" in name:
        return "1000 SFT"
    if "5000" in name:
        return "5000 SFT"
    return name.replace("qwen36_27b_", "").replace("_", " ")


def main_accuracy_rows(summary: dict[str, Any]) -> list[tuple[str, float, str]]:
    rows = [("baseline", accuracy(require(summary, "baseline", "summary"), "summary.baseline"), GRAY)]
    runs = require(summary, "runs", "summary")
    if not isinstance(runs, list):
        raise TypeError("summary.runs must be a list")
    colors = [RED, ORANGE, TEAL, BLUE]
    for index, run in enumerate(runs):
        name = str(require(run, "name", f"summary.runs[{index}]"))
        rows.append((human_run_name(name), accuracy(run, f"summary.runs[{index}]"), colors[index % len(colors)]))
    return rows


def draw_bar_chart(
    rows: list[tuple[str, float, str]],
    title: str,
    subtitle: str,
    out_path: Path,
) -> None:
    width, height = 920, 420
    left, top, right, bottom = 82, 56, 874, 320
    body: list[str] = []
    add_axes(body, left, top, right, bottom)
    slot = (right - left) / len(rows)
    bar_width = min(86, slot * 0.58)
    for index, (label, value, color) in enumerate(rows):
        center = left + slot * (index + 0.5)
        bar_x = center - bar_width / 2
        bar_y = y_scale(value, top, bottom)
        body.append(rect(bar_x, bar_y, bar_width, bottom - bar_y, color, 4))
        body.append(text(center, bar_y - 10, pct(value), size=16, fill=INK, anchor="middle", weight="700"))
        body.append(text(center, bottom + 28, label, size=15, fill=INK, anchor="middle"))
    out_path.write_text(svg_frame(width, height, title, body), encoding="utf-8")


def draw_confirmation_stress(summary: dict[str, Any], out_path: Path) -> None:
    width, height = 920, 430
    left, top, right, bottom = 92, 66, 874, 332
    body: list[str] = []
    add_axes(body, left, top, right, bottom)

    groups = []
    for key, label in [("confirmation", "confirmation"), ("stress", "stress")]:
        block = require(summary, key, f"summary.{key}")
        baseline = accuracy(require(block, "baseline", f"summary.{key}"), f"summary.{key}.baseline")
        best = accuracy(require(block, "best_run", f"summary.{key}"), f"summary.{key}.best_run")
        groups.append((label, baseline, best))

    group_slot = (right - left) / len(groups)
    bar_width = 88
    for index, (label, baseline, sft) in enumerate(groups):
        center = left + group_slot * (index + 0.5)
        for x_offset, value, color, name in [
            (-bar_width * 0.62, baseline, GRAY, "baseline"),
            (bar_width * 0.62, sft, BLUE, "5000 SFT"),
        ]:
            bar_x = center + x_offset - bar_width / 2
            bar_y = y_scale(value, top, bottom)
            body.append(rect(bar_x, bar_y, bar_width, bottom - bar_y, color, 4))
            body.append(text(center + x_offset, bar_y - 10, pct(value), size=16, anchor="middle", weight="700"))
            body.append(text(center + x_offset, bottom + 28, name, size=14, fill=MUTED, anchor="middle"))
        body.append(text(center, bottom + 56, label, size=17, fill=INK, anchor="middle", weight="700"))

    body.append(rect(650, 24, 16, 16, GRAY, 2))
    body.append(text(674, 38, "baseline", size=15, fill=MUTED))
    body.append(rect(750, 24, 16, 16, BLUE, 2))
    body.append(text(774, 38, "5000 SFT", size=15, fill=MUTED))
    out_path.write_text(svg_frame(width, height, "Confirmation and Stress Evaluation", body), encoding="utf-8")


def draw_accuracy_by_equilibria(summary: dict[str, Any], out_path: Path) -> None:
    stress = require(summary, "stress", "summary.stress")
    baseline_buckets = require(require(stress, "baseline", "summary.stress"), "accuracy_by_number_of_equilibria", "summary.stress.baseline")
    best_buckets = require(require(stress, "best_run", "summary.stress"), "accuracy_by_number_of_equilibria", "summary.stress.best_run")
    counts = [str(count) for count in range(5)]

    width, height = 980, 440
    left, top, right, bottom = 92, 66, 930, 326
    body: list[str] = []
    add_axes(body, left, top, right, bottom)

    slot = (right - left) / len(counts)
    bar_width = 50
    for index, count in enumerate(counts):
        if count not in baseline_buckets or count not in best_buckets:
            raise KeyError(f"missing equilibrium count bucket {count!r} in stress summary")
        center = left + slot * (index + 0.5)
        for x_offset, buckets, color in [
            (-bar_width * 0.58, baseline_buckets, GRAY),
            (bar_width * 0.58, best_buckets, BLUE),
        ]:
            value = float(require(buckets[count], "accuracy", f"stress bucket {count}"))
            bar_x = center + x_offset - bar_width / 2
            bar_y = y_scale(value, top, bottom)
            body.append(rect(bar_x, bar_y, bar_width, bottom - bar_y, color, 3))
            body.append(text(center + x_offset, bar_y - 8, pct(value), size=14, anchor="middle", weight="700"))
        x_label = "zero equilibria" if count == "0" else count
        body.append(text(center, bottom + 30, x_label, size=14, fill=INK, anchor="middle"))

    body.append(text(left, bottom + 66, "Number of pure-strategy Nash equilibria", size=15, fill=MUTED))
    body.append(rect(650, 24, 16, 16, GRAY, 2))
    body.append(text(674, 38, "baseline", size=15, fill=MUTED))
    body.append(rect(750, 24, 16, 16, BLUE, 2))
    body.append(text(774, 38, "5000 SFT", size=15, fill=MUTED))
    out_path.write_text(svg_frame(width, height, "Stress Accuracy by Number of Equilibria", body), encoding="utf-8")


def draw_robustness_by_variant(robustness: dict[str, Any], out_path: Path) -> None:
    baseline = require(require(robustness, "baseline", "robustness"), "accuracy_by_prompt_variant", "robustness.baseline")
    finetuned = require(require(robustness, "finetuned", "robustness"), "accuracy_by_prompt_variant", "robustness.finetuned")
    variants = sorted(baseline)

    width, height = 1080, 450
    left, top, right, bottom = 96, 66, 1030, 330
    body: list[str] = []
    add_axes(body, left, top, right, bottom)

    slot = (right - left) / len(variants)
    bar_width = min(48, slot * 0.24)
    for index, variant in enumerate(variants):
        if variant not in finetuned:
            raise KeyError(f"missing prompt variant {variant!r} in robustness finetuned summary")
        center = left + slot * (index + 0.5)
        for x_offset, row, color in [
            (-bar_width * 0.62, baseline[variant], GRAY),
            (bar_width * 0.62, finetuned[variant], BLUE),
        ]:
            value = float(require(row, "accuracy", f"robustness variant {variant}"))
            bar_x = center + x_offset - bar_width / 2
            bar_y = y_scale(value, top, bottom)
            body.append(rect(bar_x, bar_y, bar_width, bottom - bar_y, color, 3))
            body.append(text(center + x_offset, bar_y - 8, pct(value), size=13, anchor="middle", weight="700"))
        body.append(text(center, bottom + 30, variant.replace("_", " "), size=13, fill=INK, anchor="middle"))

    body.append(rect(780, 24, 16, 16, GRAY, 2))
    body.append(text(804, 38, "baseline", size=15, fill=MUTED))
    body.append(rect(880, 24, 16, 16, BLUE, 2))
    body.append(text(904, 38, "5000 SFT", size=15, fill=MUTED))
    out_path.write_text(svg_frame(width, height, "Robustness Accuracy by Prompt Variant", body), encoding="utf-8")


def adversarial_accuracy(row: dict[str, Any], run_key: str) -> float | None:
    run = require(row, run_key, f"adversarial evaluation {run_key}")
    if run.get("status") != "complete":
        return None
    return accuracy(run, f"adversarial evaluation {run_key}")


def draw_adversarial_comparison(adversarial: dict[str, Any], out_path: Path) -> None:
    evaluations = require(adversarial, "evaluations", "adversarial")
    names = ["canonical", "confirmation", "stress", "robustness"]

    width, height = 1040, 450
    left, top, right, bottom = 96, 66, 990, 330
    body: list[str] = []
    add_axes(body, left, top, right, bottom)

    slot = (right - left) / len(names)
    bar_width = 58
    for index, name in enumerate(names):
        if name not in evaluations:
            raise KeyError(f"missing adversarial evaluation {name!r}")
        row = evaluations[name]
        center = left + slot * (index + 0.5)
        for x_offset, run_key, color, label in [
            (-bar_width * 0.62, "original_5000_sft", BLUE, "5000 SFT"),
            (bar_width * 0.62, "adversarial_sft", TEAL, "adv SFT"),
        ]:
            value = adversarial_accuracy(row, run_key)
            bar_x = center + x_offset - bar_width / 2
            if value is None:
                body.append(rect(bar_x, bottom - 6, bar_width, 6, color, 3))
                body.append(text(center + x_offset, bottom - 14, "pending", size=13, anchor="middle", fill=MUTED))
                continue
            bar_y = y_scale(value, top, bottom)
            body.append(rect(bar_x, bar_y, bar_width, bottom - bar_y, color, 3))
            body.append(text(center + x_offset, bar_y - 8, pct(value), size=14, anchor="middle", weight="700"))
        body.append(text(center, bottom + 30, name, size=14, fill=INK, anchor="middle"))

    body.append(rect(720, 24, 16, 16, BLUE, 2))
    body.append(text(744, 38, "5000 SFT", size=15, fill=MUTED))
    body.append(rect(830, 24, 16, 16, TEAL, 2))
    body.append(text(854, 38, "adversarial SFT", size=15, fill=MUTED))
    out_path.write_text(svg_frame(width, height, "Adversarial SFT Comparison", body), encoding="utf-8")


def draw_seed_sweep(seed_sweep: dict[str, Any], out_path: Path) -> None:
    sizes = [int(size) for size in require(seed_sweep, "train_sizes", "seed_sweep")]
    runs = require(seed_sweep, "runs", "seed_sweep")
    statistics = require(seed_sweep, "statistics", "seed_sweep")
    baseline = accuracy(require(seed_sweep, "baseline", "seed_sweep"), "seed_sweep.baseline")

    width, height = 1040, 450
    left, top, right, bottom = 104, 66, 980, 330
    body: list[str] = []
    add_axes(body, left, top, right, bottom)

    body.append(line(left, y_scale(baseline, top, bottom), right, y_scale(baseline, top, bottom), color=GRAY, width=1.8))
    body.append(text(right - 4, y_scale(baseline, top, bottom) - 8, f"baseline {pct(baseline)}", size=14, fill=GRAY, anchor="end"))

    if len(sizes) == 1:
        x_positions = {sizes[0]: (left + right) / 2}
    else:
        x_positions = {
            size: left + index * (right - left) / (len(sizes) - 1)
            for index, size in enumerate(sizes)
        }

    points: list[tuple[float, float]] = []
    for size in sizes:
        size_key = str(size)
        x = x_positions[size]
        rows = require(runs, size_key, f"seed_sweep.runs[{size_key}]")
        if not isinstance(rows, list):
            raise TypeError(f"seed_sweep.runs[{size_key}] must be a list")
        for row in rows:
            if row.get("status") != "complete":
                continue
            value = accuracy(row, f"seed_sweep.runs[{size_key}]")
            y = y_scale(value, top, bottom)
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.0" fill="{TEAL}" opacity="0.42" />')

        stat = require(statistics, size_key, f"seed_sweep.statistics[{size_key}]")
        if stat.get("status") != "complete":
            continue
        mean_acc = float(require(require(stat, "accuracy", f"seed_sweep.statistics[{size_key}]"), "mean", f"seed_sweep.statistics[{size_key}].accuracy"))
        y = y_scale(mean_acc, top, bottom)
        points.append((x, y))
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7.0" fill="{BLUE}" />')
        body.append(text(x, y - 14, pct(mean_acc), size=15, anchor="middle", weight="700"))
        body.append(text(x, bottom + 30, str(size), size=14, fill=INK, anchor="middle"))

    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        body.append(line(x1, y1, x2, y2, color=BLUE, width=2.5))

    body.append(text((left + right) / 2, bottom + 62, "SFT training examples", size=15, fill=MUTED, anchor="middle"))
    body.append(f'<circle cx="758.0" cy="30.0" r="4.0" fill="{TEAL}" opacity="0.42" />')
    body.append(text(774, 36, "seed run", size=15, fill=MUTED))
    body.append(f'<circle cx="852.0" cy="30.0" r="7.0" fill="{BLUE}" />')
    body.append(text(868, 36, "mean", size=15, fill=MUTED))
    out_path.write_text(svg_frame(width, height, "Repeated-Seed Learning Curve", body), encoding="utf-8")


def write_figures(
    summary_path: Path = SUMMARY_PATH,
    out_dir: Path = FIGURE_DIR,
    robustness_path: Path | None = None,
    adversarial_path: Path | None = None,
    seed_sweep_path: Path | None = None,
) -> list[Path]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        out_dir / "accuracy_main.svg",
        out_dir / "accuracy_confirm_stress.svg",
        out_dir / "accuracy_by_equilibria.svg",
    ]
    draw_bar_chart(
        main_accuracy_rows(summary),
        "GT-Bench Canonical Test Accuracy",
        "Qwen/Qwen3.6-27B baseline vs LoRA SFT training-size sweep.",
        outputs[0],
    )
    draw_confirmation_stress(summary, outputs[1])
    draw_accuracy_by_equilibria(summary, outputs[2])
    if robustness_path is not None and robustness_path.exists():
        robust_out = out_dir / "robustness_by_variant.svg"
        robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
        draw_robustness_by_variant(robustness, robust_out)
        outputs.append(robust_out)
    if adversarial_path is not None and adversarial_path.exists():
        adversarial_out = out_dir / "adversarial_comparison.svg"
        adversarial = json.loads(adversarial_path.read_text(encoding="utf-8"))
        draw_adversarial_comparison(adversarial, adversarial_out)
        outputs.append(adversarial_out)
    if seed_sweep_path is not None and seed_sweep_path.exists():
        seed_sweep_out = out_dir / "seed_sweep_learning_curve.svg"
        seed_sweep = json.loads(seed_sweep_path.read_text(encoding="utf-8"))
        draw_seed_sweep(seed_sweep, seed_sweep_out)
        outputs.append(seed_sweep_out)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GT-Bench result figures as static SVG.")
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--robustness", type=Path, default=ROBUSTNESS_PATH)
    parser.add_argument("--adversarial", type=Path, default=ADVERSARIAL_PATH)
    parser.add_argument("--seed-sweep", type=Path, default=SEED_SWEEP_PATH)
    parser.add_argument("--out-dir", type=Path, default=FIGURE_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = write_figures(args.summary, args.out_dir, args.robustness, args.adversarial, args.seed_sweep)
    for output in outputs:
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
