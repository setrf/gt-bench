from generate_dataset import find_pure_nash_equilibria, make_payoffs
from score_predictions import parse_prediction


def test_prisoners_dilemma_style_one_equilibrium() -> None:
    payoffs = make_payoffs(
        (3, 3),
        (0, 5),
        (5, 0),
        (1, 1),
    )

    assert find_pure_nash_equilibria(payoffs) == [("D", "R")]


def test_coordination_game_two_equilibria() -> None:
    payoffs = make_payoffs(
        (4, 4),
        (0, 1),
        (1, 0),
        (3, 3),
    )

    assert find_pure_nash_equilibria(payoffs) == [("U", "L"), ("D", "R")]


def test_matching_pennies_style_no_pure_equilibrium() -> None:
    payoffs = make_payoffs(
        (1, 0),
        (0, 1),
        (0, 1),
        (1, 0),
    )

    assert find_pure_nash_equilibria(payoffs) == []


def test_tie_case_multiple_best_responses() -> None:
    payoffs = make_payoffs(
        (2, 2),
        (2, 2),
        (2, 2),
        (2, 2),
    )

    assert find_pure_nash_equilibria(payoffs) == [
        ("U", "L"),
        ("U", "R"),
        ("D", "L"),
        ("D", "R"),
    ]


def test_parser_common_prediction_formats() -> None:
    assert parse_prediction("(U, L) and D,R") == {("U", "L"), ("D", "R")}
    assert parse_prediction("The answer is U,L.") == {("U", "L")}
    assert parse_prediction("none") == set()
    assert parse_prediction("There is no pure Nash equilibrium.") == set()
    assert parse_prediction("") is None


def test_parser_prefers_conclusion_over_reasoning_mentions() -> None:
    prediction = (
        "1. Check (U, L): not an equilibrium.\n"
        "2. Check (U, R): this is a Nash equilibrium.\n"
        "3. Check (D, L): this is a Nash equilibrium.\n"
        "4. Check (D, R): not an equilibrium.\n\n"
        "Conclusion\nThe pure-strategy Nash equilibria are (U, R) and (D, L)."
    )

    assert parse_prediction(prediction) == {("U", "R"), ("D", "L")}
