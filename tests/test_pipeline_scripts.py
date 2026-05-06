import json

import pytest

from generate_stress_set import generate_balanced_examples
from generate_robustness_set import PROMPT_VARIANTS, generate_robustness_examples
from make_sweep_splits import write_sweep_splits
from plot_results import write_figures
from score_robustness import score_robustness
from summarize_results import exact_binomial_ci


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
