import json

import pytest

from generate_stress_set import generate_balanced_examples
from make_sweep_splits import write_sweep_splits
from plot_results import write_figures
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


def test_exact_binomial_ci_contains_observed_accuracy() -> None:
    ci = exact_binomial_ci(8, 10)

    assert ci["method"] == "clopper_pearson"
    assert ci["lower"] < 0.8 < ci["upper"]
