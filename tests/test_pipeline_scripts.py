import json

import pytest

from generate_adversarial_training import (
    combine_chat_files,
    generate_adversarial_examples,
    metadata_signature,
    to_adversarial_chat_row,
)
from make_repeated_seed_splits import generate_seed_splits
from generate_stress_set import generate_balanced_examples
from generate_robustness_set import PROMPT_VARIANTS, generate_robustness_examples
from make_sweep_splits import write_sweep_splits
from plot_results import write_figures
from score_robustness import score_robustness
from summarize_adversarial import (
    build_summary as build_adversarial_summary,
    public_adversarial_fallbacks,
    public_original_fallbacks,
)
from summarize_results import exact_binomial_ci
from summarize_seed_sweep import build_summary as build_seed_sweep_summary


def test_write_sweep_splits_uses_first_n_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    train_chat = tmp_path / "data" / "train_chat.jsonl"
    train_chat.parent.mkdir()
    rows = [{"messages": [{"role": "user", "content": str(i)}]} for i in range(5)]
    train_chat.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    config = {
        "train_sizes": [2, 4],
        "data": {
            "train_chat": str(train_chat),
            "sweep_dir": "data/sweeps",
            "sweep_template": "train_{size:04d}_chat.jsonl",
        },
    }

    outputs = write_sweep_splits(config)

    assert outputs == [
        ((tmp_path / "data" / "sweeps" / "train_0002_chat.jsonl").resolve(), 2),
        ((tmp_path / "data" / "sweeps" / "train_0004_chat.jsonl").resolve(), 4),
    ]
    first_sweep = (tmp_path / "data" / "sweeps" / "train_0002_chat.jsonl").read_text(
        encoding="utf-8"
    )
    assert first_sweep.count("\n") == 2
    assert json.loads(first_sweep.splitlines()[0]) == rows[0]


def test_generate_balanced_examples_hits_requested_counts() -> None:
    examples = generate_balanced_examples(
        per_count=2,
        seed=123,
        counts=[0, 1, 2],
        max_attempts=10_000,
    )

    counts = [
        len(example["metadata"]["pure_nash_equilibria"])  # type: ignore[index]
        for example in examples
    ]
    assert counts.count(0) == 2
    assert counts.count(1) == 2
    assert counts.count(2) == 2


def test_write_figures_generates_expected_svg_labels(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    out_dir = tmp_path / "figures"
    buckets = {
        str(count): {"total": 10, "correct": 10 if count else 3, "accuracy": 1.0 if count else 0.3}
        for count in range(5)
    }
    perfect_buckets = {
        str(count): {"total": 10, "correct": 10, "accuracy": 1.0}
        for count in range(5)
    }
    summary_path.write_text(
        json.dumps(
            {
                "baseline": {"exact_match_accuracy": 0.8},
                "runs": [
                    {"name": "qwen36_27b_sft_0250", "exact_match_accuracy": 0.5},
                    {"name": "qwen36_27b_sft_1000", "exact_match_accuracy": 0.9},
                    {"name": "qwen36_27b_sft_5000", "exact_match_accuracy": 0.99},
                ],
                "confirmation": {
                    "baseline": {"exact_match_accuracy": 0.88},
                    "best_run": {"exact_match_accuracy": 0.99},
                },
                "stress": {
                    "baseline": {
                        "exact_match_accuracy": 0.82,
                        "accuracy_by_number_of_equilibria": buckets,
                    },
                    "best_run": {
                        "exact_match_accuracy": 1.0,
                        "accuracy_by_number_of_equilibria": perfect_buckets,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    outputs = write_figures(summary_path, out_dir)

    assert {path.name for path in outputs} == {
        "accuracy_main.svg",
        "accuracy_confirm_stress.svg",
        "accuracy_by_equilibria.svg",
    }
    combined = "\n".join(path.read_text(encoding="utf-8").lower() for path in outputs)
    assert "baseline" in combined
    assert "5000" in combined
    assert "stress" in combined
    assert "zero equilibria" in combined


def test_write_figures_fails_clearly_for_missing_keys(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"runs": []}), encoding="utf-8")

    with pytest.raises(KeyError, match="missing required key 'baseline'"):
        write_figures(summary_path, tmp_path / "figures")


def test_write_figures_adds_robustness_figure_when_summary_exists(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    robustness_path = tmp_path / "robustness.json"
    out_dir = tmp_path / "figures"
    buckets = {
        str(count): {"total": 10, "correct": 10, "accuracy": 1.0}
        for count in range(5)
    }
    summary_path.write_text(
        json.dumps(
            {
                "baseline": {"exact_match_accuracy": 0.8},
                "runs": [{"name": "qwen36_27b_sft_5000", "exact_match_accuracy": 0.99}],
                "confirmation": {
                    "baseline": {"exact_match_accuracy": 0.88},
                    "best_run": {"exact_match_accuracy": 0.99},
                },
                "stress": {
                    "baseline": {
                        "exact_match_accuracy": 0.82,
                        "accuracy_by_number_of_equilibria": buckets,
                    },
                    "best_run": {
                        "exact_match_accuracy": 1.0,
                        "accuracy_by_number_of_equilibria": buckets,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    robustness_path.write_text(
        json.dumps(
            {
                "baseline": {
                    "accuracy_by_prompt_variant": {
                        "standard_table": {"accuracy": 0.8},
                        "json_payoffs": {"accuracy": 0.6},
                    }
                },
                "finetuned": {
                    "accuracy_by_prompt_variant": {
                        "standard_table": {"accuracy": 1.0},
                        "json_payoffs": {"accuracy": 0.9},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    outputs = write_figures(summary_path, out_dir, robustness_path)

    assert "robustness_by_variant.svg" in {path.name for path in outputs}
    robust_svg = (out_dir / "robustness_by_variant.svg").read_text(encoding="utf-8").lower()
    assert "json payoffs" in robust_svg


def test_write_figures_adds_adversarial_figure_when_summary_exists(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    adversarial_path = tmp_path / "adversarial.json"
    out_dir = tmp_path / "figures"
    buckets = {
        str(count): {"total": 10, "correct": 10, "accuracy": 1.0}
        for count in range(5)
    }
    summary_path.write_text(
        json.dumps(
            {
                "baseline": {"exact_match_accuracy": 0.8},
                "runs": [{"name": "qwen36_27b_sft_5000", "exact_match_accuracy": 0.99}],
                "confirmation": {
                    "baseline": {"exact_match_accuracy": 0.88},
                    "best_run": {"exact_match_accuracy": 0.99},
                },
                "stress": {
                    "baseline": {
                        "exact_match_accuracy": 0.82,
                        "accuracy_by_number_of_equilibria": buckets,
                    },
                    "best_run": {
                        "exact_match_accuracy": 1.0,
                        "accuracy_by_number_of_equilibria": buckets,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    adversarial_path.write_text(
        json.dumps(
            {
                "evaluations": {
                    name: {
                        "original_5000_sft": {
                            "status": "complete",
                            "exact_match_accuracy": 0.9,
                        },
                        "adversarial_sft": {
                            "status": "complete",
                            "exact_match_accuracy": 0.95,
                        },
                    }
                    for name in ["canonical", "confirmation", "stress", "robustness"]
                }
            }
        ),
        encoding="utf-8",
    )

    outputs = write_figures(summary_path, out_dir, adversarial_path=adversarial_path)

    assert "adversarial_comparison.svg" in {path.name for path in outputs}
    adversarial_svg = (out_dir / "adversarial_comparison.svg").read_text(
        encoding="utf-8"
    ).lower()
    assert "adversarial sft" in adversarial_svg


def test_write_figures_adds_seed_sweep_figure_when_summary_exists(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    seed_sweep_path = tmp_path / "seed_sweep.json"
    out_dir = tmp_path / "figures"
    buckets = {
        str(count): {"total": 10, "correct": 10, "accuracy": 1.0}
        for count in range(5)
    }
    summary_path.write_text(
        json.dumps(
            {
                "baseline": {"exact_match_accuracy": 0.8},
                "runs": [{"name": "qwen36_27b_sft_5000", "exact_match_accuracy": 0.99}],
                "confirmation": {
                    "baseline": {"exact_match_accuracy": 0.88},
                    "best_run": {"exact_match_accuracy": 0.99},
                },
                "stress": {
                    "baseline": {
                        "exact_match_accuracy": 0.82,
                        "accuracy_by_number_of_equilibria": buckets,
                    },
                    "best_run": {
                        "exact_match_accuracy": 1.0,
                        "accuracy_by_number_of_equilibria": buckets,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    seed_sweep_path.write_text(
        json.dumps(
            {
                "baseline": {"exact_match_accuracy": 0.8},
                "train_sizes": [250, 1000],
                "runs": {
                    "250": [
                        {"status": "complete", "seed": 1, "exact_match_accuracy": 0.7},
                        {"status": "complete", "seed": 2, "exact_match_accuracy": 0.75},
                    ],
                    "1000": [
                        {"status": "complete", "seed": 1, "exact_match_accuracy": 0.9},
                        {"status": "complete", "seed": 2, "exact_match_accuracy": 0.92},
                    ],
                },
                "statistics": {
                    "250": {"status": "complete", "accuracy": {"mean": 0.725}},
                    "1000": {"status": "complete", "accuracy": {"mean": 0.91}},
                },
            }
        ),
        encoding="utf-8",
    )

    outputs = write_figures(summary_path, out_dir, seed_sweep_path=seed_sweep_path)

    assert "seed_sweep_learning_curve.svg" in {path.name for path in outputs}
    svg = (out_dir / "seed_sweep_learning_curve.svg").read_text(encoding="utf-8").lower()
    assert "repeated-seed learning curve" in svg
    assert "baseline" in svg


def test_exact_binomial_ci_contains_observed_accuracy() -> None:
    ci = exact_binomial_ci(8, 10)

    assert ci["method"] == "clopper_pearson"
    assert ci["lower"] < 0.8 < ci["upper"]


def test_generate_robustness_examples_is_deterministic_and_balanced() -> None:
    examples = generate_robustness_examples(
        per_bucket=2,
        seed=456,
        variants=["standard_table", "compact_pairs"],
        counts=[0, 1],
        max_attempts=20_000,
    )
    examples_again = generate_robustness_examples(
        per_bucket=2,
        seed=456,
        variants=["standard_table", "compact_pairs"],
        counts=[0, 1],
        max_attempts=20_000,
    )

    assert examples == examples_again
    by_variant_count: dict[tuple[str, int], int] = {}
    matrices = set()
    for example in examples:
        metadata = example["metadata"]  # type: ignore[index]
        variant = metadata["prompt_variant"]  # type: ignore[index]
        count = metadata["equilibrium_count"]  # type: ignore[index]
        by_variant_count[(variant, count)] = by_variant_count.get((variant, count), 0) + 1
        matrices.add(tuple(tuple(cell) for cell in metadata["payoffs"].values()))  # type: ignore[index]

    assert by_variant_count == {
        ("standard_table", 0): 2,
        ("standard_table", 1): 2,
        ("compact_pairs", 0): 2,
        ("compact_pairs", 1): 2,
    }
    assert len(matrices) == len(examples)


def test_generate_adversarial_examples_is_balanced_and_excludes_matrices() -> None:
    seed_examples = generate_adversarial_examples(
        seed=123,
        variant_counts={"compact_pairs": 4},
        counts=[0, 1],
        max_attempts=20_000,
    )
    excluded = {metadata_signature(seed_examples[0])}

    examples = generate_adversarial_examples(
        seed=123,
        variant_counts={"compact_pairs": 4, "json_payoffs": 4},
        counts=[0, 1],
        exclude_signatures=excluded,
        max_attempts=20_000,
    )
    examples_again = generate_adversarial_examples(
        seed=123,
        variant_counts={"compact_pairs": 4, "json_payoffs": 4},
        counts=[0, 1],
        exclude_signatures=excluded,
        max_attempts=20_000,
    )

    assert examples == examples_again
    assert metadata_signature(examples[0]) not in excluded
    by_variant_count: dict[tuple[str, int], int] = {}
    signatures = set()
    for example in examples:
        metadata = example["metadata"]  # type: ignore[index]
        key = (metadata["prompt_variant"], metadata["equilibrium_count"])  # type: ignore[index]
        by_variant_count[key] = by_variant_count.get(key, 0) + 1
        signatures.add(metadata_signature(example))

    assert by_variant_count == {
        ("compact_pairs", 0): 2,
        ("compact_pairs", 1): 2,
        ("json_payoffs", 0): 2,
        ("json_payoffs", 1): 2,
    }
    assert len(signatures) == len(examples)


def test_adversarial_answer_only_chat_omits_reasoning() -> None:
    example = generate_adversarial_examples(
        seed=321,
        variant_counts={"answer_only": 2},
        counts=[0],
        max_attempts=20_000,
    )[0]

    chat = to_adversarial_chat_row(example)

    assistant = chat["messages"][1]["content"]  # type: ignore[index]
    assert assistant.startswith("Final answer:")
    assert "Reasoning:" not in assistant


def test_combine_chat_files_preserves_row_counts(tmp_path) -> None:
    base = tmp_path / "base.jsonl"
    supplement = tmp_path / "supplement.jsonl"
    out = tmp_path / "combined.jsonl"
    base.write_text('{"base": 1}\n{"base": 2}\n', encoding="utf-8")
    supplement.write_text('{"supplement": 1}\n', encoding="utf-8")

    total = combine_chat_files(base, supplement, out)

    assert total == 3
    assert out.read_text(encoding="utf-8").splitlines() == [
        '{"base": 1}',
        '{"base": 2}',
        '{"supplement": 1}',
    ]


def test_generate_repeated_seed_splits_writes_expected_files(tmp_path) -> None:
    outputs = generate_seed_splits(
        seeds=[11],
        sizes=[2, 4],
        train=5,
        val=1,
        out_dir=tmp_path / "repeated",
    )

    assert [(seed, size) for seed, _, size in outputs] == [(11, 2), (11, 4)]
    seed_dir = tmp_path / "repeated" / "seed_11"
    assert (seed_dir / "train.jsonl").exists()
    assert (seed_dir / "val_chat.jsonl").exists()
    assert (seed_dir / "sweeps" / "train_0002_chat.jsonl").read_text(
        encoding="utf-8"
    ).count("\n") == 2


def test_all_prompt_variants_are_configured() -> None:
    assert set(PROMPT_VARIANTS) == {
        "standard_table",
        "compact_pairs",
        "json_payoffs",
        "minimal_matrix",
        "answer_only",
    }


def test_score_robustness_groups_and_counts_missing_predictions() -> None:
    examples = generate_robustness_examples(
        per_bucket=1,
        seed=789,
        variants=["standard_table", "answer_only"],
        counts=[0, 1],
        max_attempts=20_000,
    )
    predictions = [
        {"id": examples[0]["id"], "prediction": examples[0]["answer"]},
        {"id": examples[1]["id"], "prediction": "wrong"},
        {"id": examples[2]["id"], "prediction": examples[2]["answer"]},
    ]

    report = score_robustness(examples, predictions)

    assert report["total_examples"] == 4
    assert report["num_correct"] == 2
    assert report["num_incorrect"] == 2
    assert set(report["accuracy_by_prompt_variant"]) == {"standard_table", "answer_only"}
    assert set(report["accuracy_by_number_of_equilibria"]) == {"0", "1"}
    assert len(report["failed_examples"]) == 2
    assert any(failure["prediction"] == "" for failure in report["failed_examples"])


def test_adversarial_summary_marks_missing_new_run_pending(tmp_path) -> None:
    original_path = tmp_path / "original.json"
    original_path.write_text(
        json.dumps(
            {
                "total_examples": 10,
                "exact_match_accuracy": 0.9,
                "num_correct": 9,
                "num_incorrect": 1,
                "accuracy_by_number_of_equilibria": {},
                "failed_examples": [],
            }
        ),
        encoding="utf-8",
    )

    summary = build_adversarial_summary(
        {
            "canonical": (original_path, tmp_path / "missing_canonical.json"),
            "confirmation": (original_path, tmp_path / "missing_confirmation.json"),
            "stress": (original_path, tmp_path / "missing_stress.json"),
            "robustness": (original_path, tmp_path / "missing_robustness.json"),
        }
    )

    assert summary["status"] == "pending"
    assert summary["evaluations"]["canonical"]["adversarial_sft"]["status"] == "pending"


def test_adversarial_summary_uses_public_fallbacks(tmp_path) -> None:
    results_path = tmp_path / "gt_bench_results.json"
    robustness_path = tmp_path / "robustness_results.json"
    public_row = {
        "report_path": "reports/public_original.json",
        "total_examples": 10,
        "exact_match_accuracy": 0.9,
        "num_correct": 9,
        "num_incorrect": 1,
        "accuracy_by_number_of_equilibria": {},
        "failed_examples_preview": [],
    }
    results_path.write_text(
        json.dumps(
            {
                "best_run": public_row,
                "confirmation": {"best_run": public_row},
                "stress": {"best_run": public_row},
            }
        ),
        encoding="utf-8",
    )
    robustness_path.write_text(
        json.dumps({"finetuned": {**public_row, "accuracy_by_prompt_variant": {}}}),
        encoding="utf-8",
    )

    summary = build_adversarial_summary(
        {
            "canonical": (tmp_path / "missing_canonical.json", None),
            "confirmation": (tmp_path / "missing_confirmation.json", None),
            "stress": (tmp_path / "missing_stress.json", None),
            "robustness": (tmp_path / "missing_robustness.json", None),
        },
        public_original_fallbacks(results_path, robustness_path),
    )

    assert summary["evaluations"]["canonical"]["original_5000_sft"]["status"] == "complete"
    assert (
        summary["evaluations"]["canonical"]["original_5000_sft"]["report_path"]
        == "reports/public_original.json"
    )


def test_adversarial_summary_uses_public_adversarial_fallbacks(tmp_path) -> None:
    adversarial_path = tmp_path / "adversarial_results.json"
    public_adversarial = {
        "status": "complete",
        "report_path": "reports/public_adversarial.json",
        "total_examples": 20,
        "exact_match_accuracy": 0.95,
        "num_correct": 19,
        "num_incorrect": 1,
        "accuracy_by_number_of_equilibria": {},
        "failed_examples_preview": [],
    }
    adversarial_path.write_text(
        json.dumps(
            {
                "evaluations": {
                    name: {"adversarial_sft": public_adversarial}
                    for name in ("canonical", "confirmation", "stress", "robustness")
                }
            }
        ),
        encoding="utf-8",
    )

    original = {
        "status": "complete",
        "report_path": "reports/public_original.json",
        "total_examples": 10,
        "exact_match_accuracy": 0.9,
        "num_correct": 9,
        "num_incorrect": 1,
        "accuracy_by_number_of_equilibria": {},
        "failed_examples_preview": [],
    }
    summary = build_adversarial_summary(
        {
            "canonical": (None, tmp_path / "missing_canonical.json"),
            "confirmation": (None, tmp_path / "missing_confirmation.json"),
            "stress": (None, tmp_path / "missing_stress.json"),
            "robustness": (None, tmp_path / "missing_robustness.json"),
        },
        {name: original for name in ("canonical", "confirmation", "stress", "robustness")},
        public_adversarial_fallbacks(adversarial_path),
    )

    assert summary["status"] == "complete"
    assert summary["run_name"] == "qwen36_27b_sft_5000_plus_prompt_adv500"
    assert (
        summary["evaluations"]["canonical"]["adversarial_sft"]["report_path"]
        == "reports/public_adversarial.json"
    )


def write_score_report(path, correct: int, total: int) -> None:
    path.write_text(
        json.dumps(
            {
                "total_examples": total,
                "exact_match_accuracy": correct / total,
                "num_correct": correct,
                "num_incorrect": total - correct,
                "accuracy_by_number_of_equilibria": {},
                "failed_examples": [],
            }
        ),
        encoding="utf-8",
    )


def test_seed_sweep_summary_computes_statistics_and_pairwise_deltas(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    write_score_report(baseline, correct=80, total=100)
    report_paths = {}
    for seed, scores in {
        1: {250: 70, 1000: 88, 5000: 98},
        2: {250: 72, 1000: 90, 5000: 99},
        3: {250: 74, 1000: 92, 5000: 100},
    }.items():
        for size, correct in scores.items():
            path = tmp_path / f"seed{seed}_{size}.json"
            write_score_report(path, correct=correct, total=100)
            report_paths[(seed, size)] = path

    summary = build_seed_sweep_summary(
        {"model_id": "Qwen/Qwen3.6-27B", "data": {"test": str(tmp_path / "test.jsonl")}},
        baseline,
        seeds=[1, 2, 3],
        sizes=[250, 1000, 5000],
        report_paths=report_paths,
        fallback_path=tmp_path / "missing_public.json",
    )

    assert summary["status"] == "complete"
    assert summary["statistics"]["250"]["accuracy"]["n"] == 3
    assert summary["statistics"]["250"]["mean_delta_pp_vs_baseline"] == pytest.approx(-8.0)
    comparison = summary["statistics"]["paired_comparisons"]["5000_vs_1000"]
    assert comparison["n"] == 3
    assert comparison["mean_delta_pp"] == pytest.approx(9.0)


def test_seed_sweep_summary_uses_public_baseline_and_seed_fallbacks(tmp_path) -> None:
    public_results = tmp_path / "gt_bench_results.json"
    public_seed = tmp_path / "seed_sweep_results.json"
    baseline = {
        "status": "complete",
        "report_path": "reports/public_baseline.json",
        "total_examples": 10,
        "exact_match_accuracy": 0.8,
        "num_correct": 8,
        "num_incorrect": 2,
        "accuracy_by_number_of_equilibria": {},
        "failed_examples_preview": [],
    }
    row = {
        "status": "complete",
        "seed": 1,
        "train_size": 250,
        "report_path": "reports/public_seed.json",
        "total_examples": 10,
        "exact_match_accuracy": 0.9,
        "num_correct": 9,
        "num_incorrect": 1,
        "accuracy_by_number_of_equilibria": {},
        "failed_examples_preview": [],
    }
    public_results.write_text(json.dumps({"baseline": baseline}), encoding="utf-8")
    public_seed.write_text(json.dumps({"runs": {"250": [row]}}), encoding="utf-8")

    summary = build_seed_sweep_summary(
        {"model_id": "Qwen/Qwen3.6-27B", "data": {"test": str(tmp_path / "test.jsonl")}},
        tmp_path / "missing_baseline.json",
        seeds=[1],
        sizes=[250],
        fallback_path=public_seed,
        public_results_path=public_results,
    )

    assert summary["status"] == "complete"
    assert summary["baseline"]["report_path"] == "reports/public_baseline.json"
    assert summary["runs"]["250"][0]["report_path"] == "reports/public_seed.json"
