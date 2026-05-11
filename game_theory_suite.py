from __future__ import annotations

import json
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from generate_dataset import join_phrases, to_chat_row


Payoff = tuple[int, int]
Profile = tuple[str, str]
PayoffTable = dict[Profile, Payoff]

MIXED_ROWS = ("U", "D")
MIXED_COLS = ("L", "R")
DOMINANCE_ROWS = ("A", "B", "C")
DOMINANCE_COLS = ("X", "Y", "Z")
REPEATED_STRATEGIES = ("AlwaysC", "AlwaysD", "TitForTat", "GrimTrigger")
DEFAULT_SUITE_FAMILIES = (
    "mixed_2x2",
    "dominance",
    "large_normal_form",
    "extensive_form",
    "natural_language",
    "repeated_interaction",
)


@dataclass(frozen=True)
class NormalFormGame:
    rows: tuple[str, ...]
    cols: tuple[str, ...]
    payoffs: PayoffTable


@dataclass(frozen=True)
class ExtensiveFormGame:
    payoffs: dict[tuple[str, str], Payoff]


@dataclass(frozen=True)
class RepeatedGameSpec:
    horizon: int
    p1_strategy: str
    p2_strategy: str


@dataclass(frozen=True)
class RepeatedBestResponseSpec:
    horizon: int
    opponent_strategy: str


def ordered_profiles(rows: Sequence[str], cols: Sequence[str]) -> list[Profile]:
    return [(row, col) for row in rows for col in cols]


def make_payoff_table(
    rows: Sequence[str],
    cols: Sequence[str],
    values: Sequence[int],
) -> PayoffTable:
    if len(values) != len(rows) * len(cols) * 2:
        raise ValueError("payoff value count does not match game shape")
    cells = [(values[index], values[index + 1]) for index in range(0, len(values), 2)]
    return dict(zip(ordered_profiles(rows, cols), cells))


def pure_nash_equilibria(game: NormalFormGame) -> list[Profile]:
    equilibria: list[Profile] = []
    for row, col in ordered_profiles(game.rows, game.cols):
        p1_payoff = game.payoffs[(row, col)][0]
        p2_payoff = game.payoffs[(row, col)][1]
        p1_best = max(game.payoffs[(candidate, col)][0] for candidate in game.rows)
        p2_best = max(game.payoffs[(row, candidate)][1] for candidate in game.cols)
        if p1_payoff == p1_best and p2_payoff == p2_best:
            equilibria.append((row, col))
    return equilibria


def format_profile(profile: Profile) -> str:
    return f"({profile[0]}, {profile[1]})"


def format_profiles(profiles: Sequence[Profile], none_text: str = "none") -> str:
    if not profiles:
        return none_text
    return join_phrases([format_profile(profile) for profile in profiles])


def format_normal_form_prompt(game: NormalFormGame, question: str) -> str:
    col_width = 11

    def cell(row: str, col: str) -> str:
        p1, p2 = game.payoffs[(row, col)]
        return f"({p1},{p2})"

    header = " " * 8 + "".join(f"{col:<{col_width}}" for col in game.cols)
    lines = [header]
    for row in game.rows:
        entries = "".join(f"{cell(row, col):<{col_width}}" for col in game.cols)
        lines.append(f"{row:<8}{entries}")
    return (
        f"Player 1 row strategies: {', '.join(game.rows)}. "
        f"Player 2 column strategies: {', '.join(game.cols)}. "
        "Entries are (Player 1 payoff, Player 2 payoff).\n\n"
        f"{chr(10).join(lines)}\n\n"
        f"{question}"
    )


def normal_form_metadata(game: NormalFormGame, task_family: str) -> dict[str, object]:
    return {
        "schema_version": "gt_bench_suite_v1",
        "task_family": task_family,
        "players": ["Player 1", "Player 2"],
        "actions": {
            "Player 1": list(game.rows),
            "Player 2": list(game.cols),
        },
        "rows": list(game.rows),
        "cols": list(game.cols),
        "payoffs": {
            f"{row},{col}": [game.payoffs[(row, col)][0], game.payoffs[(row, col)][1]]
            for row, col in ordered_profiles(game.rows, game.cols)
        },
    }


def build_large_normal_form_example(game: NormalFormGame, example_id: str) -> dict[str, object]:
    equilibria = pure_nash_equilibria(game)
    answer = f"The pure-strategy Nash equilibria are {format_profiles(equilibria)}."
    metadata = normal_form_metadata(game, "large_normal_form")
    metadata["pure_nash_equilibria"] = [[row, col] for row, col in equilibria]
    metadata["gold"] = {"pure_nash_equilibria": metadata["pure_nash_equilibria"]}
    metadata["difficulty"] = {
        "bucket": f"{len(game.rows)}x{len(game.cols)}_eq{len(equilibria)}",
        "shape": f"{len(game.rows)}x{len(game.cols)}",
        "equilibrium_count": len(equilibria),
    }
    return {
        "id": example_id,
        "prompt": format_normal_form_prompt(
            game,
            "Find all pure-strategy Nash equilibria. Final answer format: "
            "NE=(row, column), ... or NE=none.",
        ),
        "answer": answer,
        "solution": large_normal_form_solution(game, equilibria),
        "metadata": metadata,
    }


def large_normal_form_solution(game: NormalFormGame, equilibria: Sequence[Profile]) -> str:
    if not equilibria:
        return "Checking every cell shows no profile where both players are best responding."
    return (
        "Checking each row-column profile, the mutual best-response profiles are "
        f"{format_profiles(equilibria)}."
    )


def fully_mixed_equilibrium_2x2(game: NormalFormGame) -> dict[str, dict[str, Fraction]] | None:
    if game.rows != MIXED_ROWS or game.cols != MIXED_COLS:
        raise ValueError("fully mixed solver expects U/D rows and L/R columns")

    u1_ul = game.payoffs[("U", "L")][0]
    u1_ur = game.payoffs[("U", "R")][0]
    u1_dl = game.payoffs[("D", "L")][0]
    u1_dr = game.payoffs[("D", "R")][0]
    u2_ul = game.payoffs[("U", "L")][1]
    u2_ur = game.payoffs[("U", "R")][1]
    u2_dl = game.payoffs[("D", "L")][1]
    u2_dr = game.payoffs[("D", "R")][1]

    denom_p1 = u2_ul - u2_ur - u2_dl + u2_dr
    denom_p2 = u1_ul - u1_ur - u1_dl + u1_dr
    if denom_p1 == 0 or denom_p2 == 0:
        return None

    p1_u = Fraction(u2_dr - u2_dl, denom_p1)
    p2_l = Fraction(u1_dr - u1_ur, denom_p2)
    if not (0 < p1_u < 1 and 0 < p2_l < 1):
        return None

    return {
        "P1": {"U": p1_u, "D": 1 - p1_u},
        "P2": {"L": p2_l, "R": 1 - p2_l},
    }


def fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def build_mixed_2x2_example(game: NormalFormGame, example_id: str) -> dict[str, object]:
    equilibrium = fully_mixed_equilibrium_2x2(game)
    if equilibrium is None:
        raise ValueError("game does not have a unique fully mixed equilibrium")
    pure = pure_nash_equilibria(game)
    if pure:
        raise ValueError("mixed_2x2 examples require no pure-strategy Nash equilibrium")
    metadata = normal_form_metadata(game, "mixed_2x2")
    metadata["pure_nash_equilibria"] = [[row, col] for row, col in pure]
    metadata["mixed_nash_equilibrium"] = {
        player: {strategy: fraction_text(probability) for strategy, probability in strategies.items()}
        for player, strategies in equilibrium.items()
    }
    denominators = [
        probability.denominator
        for strategies in equilibrium.values()
        for probability in strategies.values()
    ]
    metadata["gold"] = {"mixed_nash_equilibrium": metadata["mixed_nash_equilibrium"]}
    metadata["difficulty"] = {
        "bucket": f"max_denominator_{max(denominators)}",
        "max_denominator": max(denominators),
    }
    answer = (
        "P1: U={u}, D={d}; P2: L={l}, R={r}.".format(
            u=fraction_text(equilibrium["P1"]["U"]),
            d=fraction_text(equilibrium["P1"]["D"]),
            l=fraction_text(equilibrium["P2"]["L"]),
            r=fraction_text(equilibrium["P2"]["R"]),
        )
    )
    solution = (
        "Set Player 1's row mix to make Player 2 indifferent and Player 2's column mix "
        f"to make Player 1 indifferent. The unique fully mixed equilibrium is {answer}"
    )
    return {
        "id": example_id,
        "prompt": format_normal_form_prompt(
            game,
            "This game has no pure-strategy Nash equilibrium. Find the mixed-strategy "
            "Nash equilibrium. Final answer format: P1: U=..., D=...; P2: L=..., R=....",
        ),
        "answer": answer,
        "solution": solution,
        "metadata": metadata,
    }


def strictly_dominated_rows(
    game: NormalFormGame,
    rows: Sequence[str] | None = None,
    cols: Sequence[str] | None = None,
) -> list[str]:
    active_rows = tuple(rows or game.rows)
    active_cols = tuple(cols or game.cols)
    dominated: list[str] = []
    for row in active_rows:
        for other in active_rows:
            if row == other:
                continue
            if all(game.payoffs[(other, col)][0] > game.payoffs[(row, col)][0] for col in active_cols):
                dominated.append(row)
                break
    return dominated


def strictly_dominated_cols(
    game: NormalFormGame,
    rows: Sequence[str] | None = None,
    cols: Sequence[str] | None = None,
) -> list[str]:
    active_rows = tuple(rows or game.rows)
    active_cols = tuple(cols or game.cols)
    dominated: list[str] = []
    for col in active_cols:
        for other in active_cols:
            if col == other:
                continue
            if all(game.payoffs[(row, other)][1] > game.payoffs[(row, col)][1] for row in active_rows):
                dominated.append(col)
                break
    return dominated


def iterated_elimination(game: NormalFormGame) -> dict[str, object]:
    rows = list(game.rows)
    cols = list(game.cols)
    rounds: list[dict[str, list[str]]] = []

    while True:
        dominated_rows = strictly_dominated_rows(game, rows, cols)
        dominated_cols = strictly_dominated_cols(game, rows, cols)
        if not dominated_rows and not dominated_cols:
            break
        rounds.append({"rows": dominated_rows, "cols": dominated_cols})
        rows = [row for row in rows if row not in dominated_rows]
        cols = [col for col in cols if col not in dominated_cols]

    return {
        "rounds": rounds,
        "remaining_rows": rows,
        "remaining_cols": cols,
    }


def build_dominance_example(game: NormalFormGame, example_id: str) -> dict[str, object]:
    elimination = iterated_elimination(game)
    remaining_rows = [str(row) for row in elimination["remaining_rows"]]
    remaining_cols = [str(col) for col in elimination["remaining_cols"]]
    metadata = normal_form_metadata(game, "dominance")
    metadata["iterated_elimination"] = elimination
    eliminated = sum(
        len(round_info["rows"]) + len(round_info["cols"])
        for round_info in elimination["rounds"]  # type: ignore[index]
    )
    metadata["gold"] = {
        "remaining_rows": elimination["remaining_rows"],
        "remaining_cols": elimination["remaining_cols"],
    }
    metadata["difficulty"] = {
        "bucket": f"rounds_{len(elimination['rounds'])}_eliminated_{eliminated}",
        "rounds": len(elimination["rounds"]),  # type: ignore[arg-type]
        "eliminated_strategies": eliminated,
    }
    answer = (
        f"Remaining P1 strategies: {', '.join(remaining_rows)}; "
        f"Remaining P2 strategies: {', '.join(remaining_cols)}."
    )
    solution = dominance_solution(elimination)
    return {
        "id": example_id,
        "prompt": format_normal_form_prompt(
            game,
            "Iteratively eliminate strictly dominated pure strategies for both players. "
            "Final answer format: Remaining P1 strategies: ...; Remaining P2 strategies: ....",
        ),
        "answer": answer,
        "solution": solution,
        "metadata": metadata,
    }


def dominance_solution(elimination: dict[str, object]) -> str:
    rounds = elimination["rounds"]
    if not isinstance(rounds, list) or not rounds:
        return "No strictly dominated pure strategy can be eliminated."
    parts: list[str] = []
    for index, round_info in enumerate(rounds, start=1):
        if not isinstance(round_info, dict):
            continue
        rows = ", ".join(str(row) for row in round_info.get("rows", [])) or "none"
        cols = ", ".join(str(col) for col in round_info.get("cols", [])) or "none"
        parts.append(f"Round {index}: remove P1 {{{rows}}} and P2 {{{cols}}}.")
    parts.append(
        "Remaining strategies are P1 {{{rows}}} and P2 {{{cols}}}.".format(
            rows=", ".join(str(row) for row in elimination["remaining_rows"]),
            cols=", ".join(str(col) for col in elimination["remaining_cols"]),
        )
    )
    return " ".join(parts)


def solve_extensive_form(game: ExtensiveFormGame) -> dict[str, str]:
    left_action = max(("A", "B"), key=lambda action: game.payoffs[("Left", action)][1])
    right_action = max(("A", "B"), key=lambda action: game.payoffs[("Right", action)][1])
    left_payoff = game.payoffs[("Left", left_action)][0]
    right_payoff = game.payoffs[("Right", right_action)][0]
    p1_action = "Left" if left_payoff >= right_payoff else "Right"
    path_action = left_action if p1_action == "Left" else right_action
    return {
        "p1": p1_action,
        "p2_left": left_action,
        "p2_right": right_action,
        "path": f"{p1_action}-{path_action}",
    }


def build_extensive_form_example(game: ExtensiveFormGame, example_id: str) -> dict[str, object]:
    solution = solve_extensive_form(game)
    answer = (
        f"P1={solution['p1']}; P2 after Left={solution['p2_left']}; "
        f"P2 after Right={solution['p2_right']}; Path={solution['path']}."
    )
    lines = []
    for p1_action in ("Left", "Right"):
        for p2_action in ("A", "B"):
            p1, p2 = game.payoffs[(p1_action, p2_action)]
            lines.append(f"- If P1 chooses {p1_action} and P2 chooses {p2_action}: ({p1},{p2})")
    return {
        "id": example_id,
        "prompt": (
            "Perfect-information sequential game. P1 first chooses Left or Right. "
            "After observing P1's choice, P2 chooses A or B at that reached node. "
            "Terminal payoffs are (P1 payoff, P2 payoff):\n"
            f"{chr(10).join(lines)}\n\n"
            "Use backward induction. Final answer format: P1=...; P2 after Left=...; "
            "P2 after Right=...; Path=...."
        ),
        "answer": answer,
        "solution": (
            "Backward induction selects P2's best action at each node, then P1's best "
            f"initial branch. {answer}"
        ),
        "metadata": {
            "schema_version": "gt_bench_suite_v1",
            "task_family": "extensive_form",
            "players": ["Player 1", "Player 2"],
            "actions": {
                "Player 1": ["Left", "Right"],
                "Player 2": ["A", "B"],
            },
            "payoffs": {
                f"{p1},{p2}": [game.payoffs[(p1, p2)][0], game.payoffs[(p1, p2)][1]]
                for p1 in ("Left", "Right")
                for p2 in ("A", "B")
            },
            "subgame_perfect_equilibrium": solution,
            "gold": {"subgame_perfect_equilibrium": solution},
            "difficulty": {
                "bucket": f"path_{solution['p1'].lower()}",
                "path_branch": solution["p1"],
            },
        },
    }


def build_natural_language_example(game: NormalFormGame, example_id: str) -> dict[str, object]:
    equilibria = pure_nash_equilibria(game)
    row_1, row_2 = game.rows
    col_1, col_2 = game.cols
    sentences = [
        "A startup chooses Invest or Wait. A rival chooses Enter or StayOut.",
        (
            f"If the startup chooses {row_1} and the rival chooses {col_1}, "
            f"their payoffs are {game.payoffs[(row_1, col_1)][0]} and "
            f"{game.payoffs[(row_1, col_1)][1]}."
        ),
        (
            f"If the startup chooses {row_1} and the rival chooses {col_2}, "
            f"their payoffs are {game.payoffs[(row_1, col_2)][0]} and "
            f"{game.payoffs[(row_1, col_2)][1]}."
        ),
        (
            f"If the startup chooses {row_2} and the rival chooses {col_1}, "
            f"their payoffs are {game.payoffs[(row_2, col_1)][0]} and "
            f"{game.payoffs[(row_2, col_1)][1]}."
        ),
        (
            f"If the startup chooses {row_2} and the rival chooses {col_2}, "
            f"their payoffs are {game.payoffs[(row_2, col_2)][0]} and "
            f"{game.payoffs[(row_2, col_2)][1]}."
        ),
    ]
    metadata = normal_form_metadata(game, "natural_language")
    metadata["pure_nash_equilibria"] = [[row, col] for row, col in equilibria]
    metadata["gold"] = {"pure_nash_equilibria": metadata["pure_nash_equilibria"]}
    metadata["difficulty"] = {
        "bucket": f"story_eq{len(equilibria)}",
        "equilibrium_count": len(equilibria),
    }
    return {
        "id": example_id,
        "prompt": (
            "Read the game description and identify all pure-strategy Nash equilibria.\n\n"
            f"{' '.join(sentences)}\n\n"
            "Final answer format: NE=(startup action, rival action), ... or NE=none."
        ),
        "answer": f"The pure-strategy Nash equilibria are {format_profiles(equilibria)}.",
        "solution": large_normal_form_solution(game, equilibria),
        "metadata": metadata,
    }


def repeated_action(strategy: str, own_history: Sequence[str], other_history: Sequence[str]) -> str:
    if strategy == "AlwaysC":
        return "C"
    if strategy == "AlwaysD":
        return "D"
    if strategy == "TitForTat":
        return "C" if not other_history else other_history[-1]
    if strategy == "GrimTrigger":
        return "D" if "D" in other_history else "C"
    raise ValueError(f"unknown repeated-game strategy: {strategy}")


def simulate_repeated_pd(p1_strategy: str, p2_strategy: str, horizon: int) -> tuple[int, int]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    payoffs = {
        ("C", "C"): (3, 3),
        ("C", "D"): (0, 5),
        ("D", "C"): (5, 0),
        ("D", "D"): (1, 1),
    }
    p1_history: list[str] = []
    p2_history: list[str] = []
    total_p1 = 0
    total_p2 = 0
    for _ in range(horizon):
        p1_action = repeated_action(p1_strategy, p1_history, p2_history)
        p2_action = repeated_action(p2_strategy, p2_history, p1_history)
        p1_payoff, p2_payoff = payoffs[(p1_action, p2_action)]
        total_p1 += p1_payoff
        total_p2 += p2_payoff
        p1_history.append(p1_action)
        p2_history.append(p2_action)
    return total_p1, total_p2


def solve_repeated_simulation(spec: RepeatedGameSpec) -> dict[str, int]:
    p1_payoff, p2_payoff = simulate_repeated_pd(
        spec.p1_strategy,
        spec.p2_strategy,
        spec.horizon,
    )
    return {
        "p1_payoff": p1_payoff,
        "p2_payoff": p2_payoff,
    }


def solve_repeated_best_response(spec: RepeatedBestResponseSpec) -> dict[str, object]:
    totals = {
        strategy: simulate_repeated_pd(strategy, spec.opponent_strategy, spec.horizon)[0]
        for strategy in REPEATED_STRATEGIES
    }
    best_strategy = max(
        REPEATED_STRATEGIES,
        key=lambda strategy: (
            totals[strategy],
            -REPEATED_STRATEGIES.index(strategy),
        ),
    )
    return {
        "best_strategy": best_strategy,
        "best_payoff": totals[best_strategy],
        "totals": totals,
    }


def build_repeated_example(spec: RepeatedGameSpec, example_id: str) -> dict[str, object]:
    result = solve_repeated_simulation(spec)
    answer = f"P1 payoff: {result['p1_payoff']}; P2 payoff: {result['p2_payoff']}."
    return {
        "id": example_id,
        "prompt": (
            "Finite repeated prisoner's dilemma. Stage payoffs are CC=(3,3), CD=(0,5), "
            "DC=(5,0), DD=(1,1), where the first letter is P1's action. "
            f"The game lasts {spec.horizon} rounds. P1 uses {spec.p1_strategy}; "
            f"P2 uses {spec.p2_strategy}. Compute total payoffs over all rounds. "
            "Final answer format: P1 payoff: ...; P2 payoff: ...."
        ),
        "answer": answer,
        "solution": f"Simulating the repeated interaction gives {answer}",
        "metadata": {
            "schema_version": "gt_bench_suite_v1",
            "task_family": "repeated_interaction",
            "repeated_task": "simulation",
            "players": ["Player 1", "Player 2"],
            "actions": {"both": ["C", "D"]},
            "horizon": spec.horizon,
            "p1_strategy": spec.p1_strategy,
            "p2_strategy": spec.p2_strategy,
            "simulation": result,
            "gold": {"simulation": result},
            "difficulty": {
                "bucket": f"horizon_{spec.horizon}",
                "horizon": spec.horizon,
                "strategy_pair": f"{spec.p1_strategy}_vs_{spec.p2_strategy}",
            },
        },
    }


def build_repeated_best_response_example(
    spec: RepeatedBestResponseSpec,
    example_id: str,
) -> dict[str, object]:
    result = solve_repeated_best_response(spec)
    totals = result["totals"]
    if not isinstance(totals, dict):
        raise ValueError("best-response totals are malformed")
    answer = f"Best P1 strategy: {result['best_strategy']}; P1 payoff: {result['best_payoff']}."
    total_text = ", ".join(f"{strategy}={totals[strategy]}" for strategy in REPEATED_STRATEGIES)
    return {
        "id": example_id,
        "prompt": (
            "Finite repeated prisoner's dilemma. Stage payoffs are CC=(3,3), CD=(0,5), "
            "DC=(5,0), DD=(1,1), where the first letter is P1's action. "
            f"The game lasts {spec.horizon} rounds. P2 uses {spec.opponent_strategy}. "
            "P1 must choose one policy from AlwaysC, AlwaysD, TitForTat, GrimTrigger. "
            "Which P1 policy gives the highest total P1 payoff? "
            "Final answer format: Best P1 strategy: ...; P1 payoff: ...."
        ),
        "answer": answer,
        "solution": f"Simulating each candidate gives {total_text}. {answer}",
        "metadata": {
            "schema_version": "gt_bench_suite_v1",
            "task_family": "repeated_interaction",
            "repeated_task": "best_response",
            "players": ["Player 1", "Player 2"],
            "actions": {"both": ["C", "D"]},
            "horizon": spec.horizon,
            "opponent_strategy": spec.opponent_strategy,
            "candidate_strategies": list(REPEATED_STRATEGIES),
            "best_response": result,
            "gold": {"best_response": result},
            "difficulty": {
                "bucket": f"best_response_horizon_{spec.horizon}",
                "horizon": spec.horizon,
                "opponent_strategy": spec.opponent_strategy,
            },
        },
    }


def unique_signature(row: dict[str, object]) -> tuple[object, ...]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return (row.get("id"),)
    task = str(metadata.get("task_family"))
    if "payoffs" in metadata:
        payoffs = metadata["payoffs"]
        if isinstance(payoffs, dict):
            return (task, tuple(sorted((key, tuple(value)) for key, value in payoffs.items())))  # type: ignore[arg-type]
    return (task, json.dumps(metadata, sort_keys=True))


def random_game(
    rng: random.Random,
    rows: Sequence[str],
    cols: Sequence[str],
    min_payoff: int = 0,
    max_payoff: int = 9,
) -> NormalFormGame:
    values = [rng.randint(min_payoff, max_payoff) for _ in range(len(rows) * len(cols) * 2)]
    return NormalFormGame(tuple(rows), tuple(cols), make_payoff_table(rows, cols, values))


def generate_family_examples(
    family: str,
    count: int,
    rng: random.Random,
    max_attempts: int,
) -> list[dict[str, object]]:
    if count < 0:
        raise ValueError("count must be nonnegative")
    examples: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    attempts = 0

    while len(examples) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"hit max_attempts={max_attempts} while generating {family}")

        example = maybe_generate_family_example(family, len(examples) + 1, rng)
        if example is None:
            continue
        signature = unique_signature(example)
        if signature in seen:
            continue
        seen.add(signature)
        examples.append(example)

    return examples


def maybe_generate_family_example(
    family: str,
    index: int,
    rng: random.Random,
) -> dict[str, object] | None:
    if family == "mixed_2x2":
        game = random_game(rng, MIXED_ROWS, MIXED_COLS)
        if pure_nash_equilibria(game):
            return None
        if fully_mixed_equilibrium_2x2(game) is None:
            return None
        return build_mixed_2x2_example(game, f"mixed_2x2_{index:04d}")

    if family == "dominance":
        game = random_game(rng, DOMINANCE_ROWS, DOMINANCE_COLS)
        elimination = iterated_elimination(game)
        if not elimination["rounds"]:
            return None
        return build_dominance_example(game, f"dominance_{index:04d}")

    if family == "large_normal_form":
        rows = ("A", "B") if index % 2 else ("A", "B", "C")
        cols = ("X", "Y", "Z")
        game = random_game(rng, rows, cols)
        return build_large_normal_form_example(game, f"large_normal_form_{index:04d}")

    if family == "extensive_form":
        values = [rng.randint(0, 9) for _ in range(8)]
        payoffs = {
            ("Left", "A"): (values[0], values[1]),
            ("Left", "B"): (values[2], values[3]),
            ("Right", "A"): (values[4], values[5]),
            ("Right", "B"): (values[6], values[7]),
        }
        game = ExtensiveFormGame(payoffs)
        solution = solve_extensive_form(game)
        if game.payoffs[("Left", "A")][1] == game.payoffs[("Left", "B")][1]:
            return None
        if game.payoffs[("Right", "A")][1] == game.payoffs[("Right", "B")][1]:
            return None
        left_payoff = game.payoffs[("Left", solution["p2_left"])][0]
        right_payoff = game.payoffs[("Right", solution["p2_right"])][0]
        if left_payoff == right_payoff:
            return None
        return build_extensive_form_example(game, f"extensive_form_{index:04d}")

    if family == "natural_language":
        game = random_game(rng, ("Invest", "Wait"), ("Enter", "StayOut"))
        return build_natural_language_example(game, f"natural_language_{index:04d}")

    if family == "repeated_interaction":
        if index % 4 == 1:
            spec = RepeatedBestResponseSpec(
                horizon=rng.randint(3, 20),
                opponent_strategy=rng.choice(REPEATED_STRATEGIES),
            )
            return build_repeated_best_response_example(spec, f"repeated_interaction_{index:04d}")
        spec = RepeatedGameSpec(
            horizon=rng.randint(3, 20),
            p1_strategy=rng.choice(REPEATED_STRATEGIES),
            p2_strategy=rng.choice(REPEATED_STRATEGIES),
        )
        return build_repeated_example(spec, f"repeated_interaction_{index:04d}")

    raise ValueError(f"unknown suite family: {family}")


def generate_suite_examples(
    per_family: int,
    seed: int,
    families: Sequence[str],
    max_attempts: int = 5_000_000,
) -> list[dict[str, object]]:
    if per_family <= 0:
        raise ValueError("per_family must be positive")
    if not families:
        raise ValueError("at least one family is required")
    rng = random.Random(seed)
    examples: list[dict[str, object]] = []
    for family in families:
        examples.extend(generate_family_examples(family, per_family, rng, max_attempts))
    return examples


def to_suite_chat_row(example: dict[str, object]) -> dict[str, object]:
    return to_chat_row(example)


def add_split_metadata(example: dict[str, object], split: str) -> dict[str, object]:
    row = dict(example)
    metadata = dict(row["metadata"])  # type: ignore[arg-type]
    metadata["split"] = split
    row["metadata"] = metadata
    return row


def split_suite_examples(
    train_per_family: int,
    val_per_family: int,
    test_per_family: int,
    seed: int,
    families: Sequence[str] = DEFAULT_SUITE_FAMILIES,
    max_attempts: int = 5_000_000,
) -> dict[str, list[dict[str, object]]]:
    total_per_family = train_per_family + val_per_family + test_per_family
    if total_per_family <= 0:
        raise ValueError("at least one split size must be positive")
    rows = generate_suite_examples(
        per_family=total_per_family,
        seed=seed,
        families=families,
        max_attempts=max_attempts,
    )
    splits = {"train": [], "val": [], "test": []}
    by_family: dict[str, list[dict[str, object]]] = {family: [] for family in families}
    for row in rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"row {row.get('id')} is missing metadata")
        by_family[str(metadata["task_family"])].append(row)

    for family in families:
        family_rows = by_family[family]
        split_points = {
            "train": family_rows[:train_per_family],
            "val": family_rows[train_per_family : train_per_family + val_per_family],
            "test": family_rows[train_per_family + val_per_family :],
        }
        for split_name, split_rows in split_points.items():
            splits[split_name].extend(add_split_metadata(row, split_name) for row in split_rows)
    return splits
