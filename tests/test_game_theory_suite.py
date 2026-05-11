import json

from game_theory_suite import (
    ExtensiveFormGame,
    NormalFormGame,
    RepeatedBestResponseSpec,
    RepeatedGameSpec,
    build_dominance_example,
    build_large_normal_form_example,
    build_mixed_2x2_example,
    build_repeated_best_response_example,
    fully_mixed_equilibrium_2x2,
    generate_suite_examples,
    iterated_elimination,
    make_payoff_table,
    pure_nash_equilibria,
    simulate_repeated_pd,
    solve_extensive_form,
    solve_repeated_best_response,
    solve_repeated_simulation,
    split_suite_examples,
)
from run_suite_baselines import build_suite_summary
from score_suite import score_suite


def test_large_normal_form_solver_handles_2x3_games() -> None:
    game = NormalFormGame(
        ("A", "B"),
        ("X", "Y", "Z"),
        make_payoff_table(
            ("A", "B"),
            ("X", "Y", "Z"),
            [
                1,
                1,
                2,
                4,
                0,
                2,
                3,
                0,
                5,
                5,
                4,
                1,
            ],
        ),
    )

    assert pure_nash_equilibria(game) == [("B", "Y")]


def test_mixed_2x2_solver_returns_exact_fractions() -> None:
    game = NormalFormGame(
        ("U", "D"),
        ("L", "R"),
        {
            ("U", "L"): (1, -1),
            ("U", "R"): (-1, 1),
            ("D", "L"): (-1, 1),
            ("D", "R"): (1, -1),
        },
    )

    equilibrium = fully_mixed_equilibrium_2x2(game)

    assert equilibrium is not None
    assert str(equilibrium["P1"]["U"]) == "1/2"
    assert str(equilibrium["P1"]["D"]) == "1/2"
    assert str(equilibrium["P2"]["L"]) == "1/2"
    assert str(equilibrium["P2"]["R"]) == "1/2"


def test_iterated_elimination_removes_strictly_dominated_strategies() -> None:
    game = NormalFormGame(
        ("A", "B", "C"),
        ("X", "Y", "Z"),
        {
            ("A", "X"): (0, 0),
            ("A", "Y"): (0, 1),
            ("A", "Z"): (0, 0),
            ("B", "X"): (2, 2),
            ("B", "Y"): (2, 3),
            ("B", "Z"): (2, 2),
            ("C", "X"): (1, 1),
            ("C", "Y"): (1, 4),
            ("C", "Z"): (1, 1),
        },
    )

    elimination = iterated_elimination(game)

    assert elimination["rounds"] == [{"rows": ["A", "C"], "cols": ["X", "Z"]}]
    assert elimination["remaining_rows"] == ["B"]
    assert elimination["remaining_cols"] == ["Y"]


def test_extensive_form_solver_uses_backward_induction() -> None:
    game = ExtensiveFormGame(
        {
            ("Left", "A"): (2, 1),
            ("Left", "B"): (4, 5),
            ("Right", "A"): (7, 3),
            ("Right", "B"): (1, 2),
        }
    )

    assert solve_extensive_form(game) == {
        "p1": "Right",
        "p2_left": "B",
        "p2_right": "A",
        "path": "Right-A",
    }


def test_repeated_prisoners_dilemma_simulation_tracks_history() -> None:
    assert simulate_repeated_pd("TitForTat", "AlwaysD", 4) == (3, 8)
    assert solve_repeated_simulation(
        RepeatedGameSpec(horizon=4, p1_strategy="GrimTrigger", p2_strategy="TitForTat")
    ) == {"p1_payoff": 12, "p2_payoff": 12}


def test_repeated_best_response_scores_policy_choice() -> None:
    spec = RepeatedBestResponseSpec(horizon=4, opponent_strategy="AlwaysC")
    result = solve_repeated_best_response(spec)

    assert result["best_strategy"] == "AlwaysD"
    assert result["best_payoff"] == 20

    example = build_repeated_best_response_example(spec, "repeated_best")
    report = score_suite([example], [{"id": "repeated_best", "prediction": example["answer"]}])

    assert report["exact_match_accuracy"] == 1.0


def test_suite_generation_is_deterministic_and_covers_all_families() -> None:
    examples = generate_suite_examples(per_family=2, seed=2026, families=[
        "mixed_2x2",
        "dominance",
        "large_normal_form",
        "extensive_form",
        "natural_language",
        "repeated_interaction",
    ])
    examples_again = generate_suite_examples(per_family=2, seed=2026, families=[
        "mixed_2x2",
        "dominance",
        "large_normal_form",
        "extensive_form",
        "natural_language",
        "repeated_interaction",
    ])

    assert examples == examples_again
    by_family: dict[str, int] = {}
    for example in examples:
        family = example["metadata"]["task_family"]  # type: ignore[index]
        by_family[family] = by_family.get(family, 0) + 1
        assert example["metadata"]["schema_version"] == "gt_bench_suite_v1"  # type: ignore[index]
        assert "gold" in example["metadata"]  # type: ignore[operator]
        assert "difficulty" in example["metadata"]  # type: ignore[operator]

    assert by_family == {
        "mixed_2x2": 2,
        "dominance": 2,
        "large_normal_form": 2,
        "extensive_form": 2,
        "natural_language": 2,
        "repeated_interaction": 2,
    }


def test_suite_scorer_accepts_gold_answers_and_rejects_wrong_answers() -> None:
    games = [
        build_mixed_2x2_example(
            NormalFormGame(
                ("U", "D"),
                ("L", "R"),
                {
                    ("U", "L"): (1, -1),
                    ("U", "R"): (-1, 1),
                    ("D", "L"): (-1, 1),
                    ("D", "R"): (1, -1),
                },
            ),
            "mixed",
        ),
        build_large_normal_form_example(
            NormalFormGame(
                ("A", "B"),
                ("X", "Y"),
                {
                    ("A", "X"): (3, 3),
                    ("A", "Y"): (0, 5),
                    ("B", "X"): (5, 0),
                    ("B", "Y"): (1, 1),
                },
            ),
            "large",
        ),
        build_dominance_example(
            NormalFormGame(
                ("A", "B"),
                ("X", "Y"),
                {
                    ("A", "X"): (0, 0),
                    ("A", "Y"): (0, 0),
                    ("B", "X"): (1, 1),
                    ("B", "Y"): (1, 1),
                },
            ),
            "dominance",
        ),
    ]
    predictions = [{"id": row["id"], "prediction": row["answer"]} for row in games]

    report = score_suite(games, predictions)

    assert report["exact_match_accuracy"] == 1.0

    wrong_predictions = json.loads(json.dumps(predictions))
    wrong_predictions[0]["prediction"] = "P1: U=1, D=0; P2: L=1, R=0."

    wrong_report = score_suite(games, wrong_predictions)

    assert wrong_report["num_correct"] == 2
    assert wrong_report["failed_examples"][0]["id"] == "mixed"


def test_suite_scorer_round_trips_all_generated_families() -> None:
    examples = generate_suite_examples(per_family=1, seed=11, families=[
        "mixed_2x2",
        "dominance",
        "large_normal_form",
        "extensive_form",
        "natural_language",
        "repeated_interaction",
    ])
    predictions = [{"id": row["id"], "prediction": row["answer"]} for row in examples]

    report = score_suite(examples, predictions)

    assert report["exact_match_accuracy"] == 1.0
    assert set(report["accuracy_by_task_family"]) == {
        "mixed_2x2",
        "dominance",
        "large_normal_form",
        "extensive_form",
        "natural_language",
        "repeated_interaction",
    }
    assert report["accuracy_by_difficulty"]


def test_split_suite_examples_balances_every_family_by_split() -> None:
    splits = split_suite_examples(
        train_per_family=2,
        val_per_family=1,
        test_per_family=3,
        seed=17,
    )

    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 12,
        "val": 6,
        "test": 18,
    }
    for split_name, rows in splits.items():
        counts: dict[str, int] = {}
        for row in rows:
            metadata = row["metadata"]  # type: ignore[index]
            assert metadata["split"] == split_name  # type: ignore[index]
            family = metadata["task_family"]  # type: ignore[index]
            counts[family] = counts.get(family, 0) + 1
        expected_count = {"train": 2, "val": 1, "test": 3}[split_name]
        assert set(counts.values()) == {expected_count}


def test_suite_baseline_summary_skips_public_model_reports_for_custom_splits(tmp_path) -> None:
    splits = split_suite_examples(
        train_per_family=1,
        val_per_family=0,
        test_per_family=1,
        seed=19,
    )
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    train_path.write_text(
        "\n".join(json.dumps(row) for row in splits["train"]) + "\n",
        encoding="utf-8",
    )
    test_path.write_text(
        "\n".join(json.dumps(row) for row in splits["test"]) + "\n",
        encoding="utf-8",
    )

    summary = build_suite_summary(
        gold_path=test_path,
        train_path=train_path,
        seed=19,
        pred_dir=tmp_path / "predictions",
    )

    assert summary["status"] == "baseline_only"
    assert summary["baselines"]["oracle"]["exact_match_accuracy"] == 1.0
    assert summary["model_evaluations"] == {}
    assert (tmp_path / "predictions" / "oracle.jsonl").exists()
