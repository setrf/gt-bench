from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Sequence


SUMMARY_PATH = Path("reports/gt_bench_results.json")
ROBUSTNESS_PATH = Path("reports/robustness_results.json")
ADVERSARIAL_PATH = Path("reports/adversarial_results.json")
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
        body.append(text(left - 12, y + 5, f"{tick}%", size=12, fill=MUTED, anchor="end"))
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
    width, height = 920, 520
    left, top, right, bottom = 82, 104, 874, 410
    body: list[str] = [
        text(40, 46, title, size=24, weight="700"),
        text(40, 74, subtitle, size=14, fill=MUTED),
    ]
    add_axes(body, left, top, right, bottom)
    slot = (right - left) / len(rows)
    bar_width = min(86, slot * 0.58)
    for index, (label, value, color) in enumerate(rows):
        center = left + slot * (index + 0.5)
        bar_x = center - bar_width / 2
        bar_y = y_scale(value, top, bottom)
        body.append(rect(bar_x, bar_y, bar_width, bottom - bar_y, color, 4))
        body.append(text(center, bar_y - 10, pct(value), size=14, fill=INK, anchor="middle", weight="700"))
        body.append(text(center, bottom + 28, label, size=13, fill=INK, anchor="middle"))
    out_path.write_text(svg_frame(width, height, title, body), encoding="utf-8")


def draw_confirmation_stress(summary: dict[str, Any], out_path: Path) -> None:
    width, height = 920, 520
    left, top, right, bottom = 92, 108, 874, 404
    body: list[str] = [
        text(40, 46, "Confirmation and Stress Evaluation", size=24, weight="700"),
        text(40, 74, "Same 5000-example LoRA checkpoint, evaluated without additional training.", size=14, fill=MUTED),
    ]
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
            body.append(text(center + x_offset, bar_y - 10, pct(value), size=14, anchor="middle", weight="700"))
            body.append(text(center + x_offset, bottom + 28, name, size=12, fill=MUTED, anchor="middle"))
        body.append(text(center, bottom + 56, label, size=16, fill=INK, anchor="middle", weight="700"))

    body.append(rect(650, 42, 16, 16, GRAY, 2))
    body.append(text(674, 55, "baseline", size=13, fill=MUTED))
    body.append(rect(750, 42, 16, 16, BLUE, 2))
    body.append(text(774, 55, "5000 SFT", size=13, fill=MUTED))
    out_path.write_text(svg_frame(width, height, "Confirmation and Stress Evaluation", body), encoding="utf-8")


def draw_accuracy_by_equilibria(summary: dict[str, Any], out_path: Path) -> None:
    stress = require(summary, "stress", "summary.stress")
    baseline_buckets = require(require(stress, "baseline", "summary.stress"), "accuracy_by_number_of_equilibria", "summary.stress.baseline")
    best_buckets = require(require(stress, "best_run", "summary.stress"), "accuracy_by_number_of_equilibria", "summary.stress.best_run")
    counts = [str(count) for count in range(5)]

    width, height = 980, 540
    left, top, right, bottom = 92, 120, 930, 410
    body: list[str] = [
        text(40, 46, "Stress Accuracy by Number of Equilibria", size=24, weight="700"),
        text(40, 74, "Balanced stress set: 50 examples in each bucket. Zero equilibria is the brittle baseline case.", size=14, fill=MUTED),
    ]
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
            body.append(text(center + x_offset, bar_y - 8, pct(value), size=12, anchor="middle", weight="700"))
        x_label = "zero equilibria" if count == "0" else count
        body.append(text(center, bottom + 30, x_label, size=12, fill=INK, anchor="middle"))

    body.append(text(left, bottom + 66, "Number of pure-strategy Nash equilibria", size=13, fill=MUTED))
    body.append(rect(650, 44, 16, 16, GRAY, 2))
    body.append(text(674, 57, "baseline", size=13, fill=MUTED))
    body.append(rect(750, 44, 16, 16, BLUE, 2))
    body.append(text(774, 57, "5000 SFT", size=13, fill=MUTED))
    out_path.write_text(svg_frame(width, height, "Stress Accuracy by Number of Equilibria", body), encoding="utf-8")


def draw_robustness_by_variant(robustness: dict[str, Any], out_path: Path) -> None:
    baseline = require(require(robustness, "baseline", "robustness"), "accuracy_by_prompt_variant", "robustness.baseline")
    finetuned = require(require(robustness, "finetuned", "robustness"), "accuracy_by_prompt_variant", "robustness.finetuned")
    variants = sorted(baseline)

    width, height = 1080, 560
    left, top, right, bottom = 96, 116, 1030, 410
    body: list[str] = [
        text(40, 46, "Robustness Accuracy by Prompt Variant", size=24, weight="700"),
        text(40, 74, "Same task, different prompt surfaces. Higher is better.", size=14, fill=MUTED),
    ]
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
            body.append(text(center + x_offset, bar_y - 8, pct(value), size=11, anchor="middle", weight="700"))
        body.append(text(center, bottom + 30, variant.replace("_", " "), size=11, fill=INK, anchor="middle"))

    body.append(rect(780, 44, 16, 16, GRAY, 2))
    body.append(text(804, 57, "baseline", size=13, fill=MUTED))
    body.append(rect(880, 44, 16, 16, BLUE, 2))
    body.append(text(904, 57, "5000 SFT", size=13, fill=MUTED))
    out_path.write_text(svg_frame(width, height, "Robustness Accuracy by Prompt Variant", body), encoding="utf-8")


def adversarial_accuracy(row: dict[str, Any], run_key: str) -> float | None:
    run = require(row, run_key, f"adversarial evaluation {run_key}")
    if run.get("status") != "complete":
        return None
    return accuracy(run, f"adversarial evaluation {run_key}")


def draw_adversarial_comparison(adversarial: dict[str, Any], out_path: Path) -> None:
    evaluations = require(adversarial, "evaluations", "adversarial")
    names = ["canonical", "confirmation", "stress", "robustness"]

    width, height = 1040, 560
    left, top, right, bottom = 96, 122, 990, 410
    body: list[str] = [
        text(40, 46, "Adversarial SFT Comparison", size=24, weight="700"),
        text(
            40,
            74,
            "Original 5000-example SFT vs prompt-adversarial SFT. Pending bars mark runs not yet completed.",
            size=14,
            fill=MUTED,
        ),
    ]
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
                body.append(text(center + x_offset, bottom - 14, "pending", size=11, anchor="middle", fill=MUTED))
                continue
            bar_y = y_scale(value, top, bottom)
            body.append(rect(bar_x, bar_y, bar_width, bottom - bar_y, color, 3))
            body.append(text(center + x_offset, bar_y - 8, pct(value), size=12, anchor="middle", weight="700"))
        body.append(text(center, bottom + 30, name, size=12, fill=INK, anchor="middle"))

    body.append(rect(720, 44, 16, 16, BLUE, 2))
    body.append(text(744, 57, "5000 SFT", size=13, fill=MUTED))
    body.append(rect(830, 44, 16, 16, TEAL, 2))
    body.append(text(854, 57, "adversarial SFT", size=13, fill=MUTED))
    out_path.write_text(svg_frame(width, height, "Adversarial SFT Comparison", body), encoding="utf-8")


def write_figures(
    summary_path: Path = SUMMARY_PATH,
    out_dir: Path = FIGURE_DIR,
    robustness_path: Path | None = None,
    adversarial_path: Path | None = None,
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
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GT-Bench result figures as static SVG.")
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--robustness", type=Path, default=ROBUSTNESS_PATH)
    parser.add_argument("--adversarial", type=Path, default=ADVERSARIAL_PATH)
    parser.add_argument("--out-dir", type=Path, default=FIGURE_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = write_figures(args.summary, args.out_dir, args.robustness, args.adversarial)
    for output in outputs:
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
