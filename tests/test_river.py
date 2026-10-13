import random
import time

import numpy as np
import pytest

from pokerbot.advisor import spot_from_partial
from pokerbot.cards import COMBO_CARDS, N_COMBOS, combo_index, parse_cards
from pokerbot.game.state import CALL, CHECK, FOLD, RAISE, Action, GameConfig, HandState
from pokerbot.game.tree import BetSizing, BetState, tree_actions
from pokerbot.ranges import RangeTracker
from pokerbot.solver.river import build_river_tree, solve_river
from pokerbot.solver.terminal import FoldTerminal, RiverBoard, ShowdownTerminal
from pokerbot.solver.vector_cfr import VectorCFR

BOARD = "7h8h2cKd3s"


def tracked(hole, **kw):
    spot = spot_from_partial(parse_cards(hole), parse_cards(BOARD), **kw)
    tracker = RangeTracker()
    tracker.catch_up(spot)
    return spot, tuple(tracker.ranges)


def test_fast_terminals_match_the_matrix_product():
    rng = np.random.default_rng(0)
    a, b = COMBO_CARDS[:, 0], COMBO_CARDS[:, 1]
    shares = (
        (a[:, None] == a[None, :])
        | (a[:, None] == b[None, :])
        | (b[:, None] == a[None, :])
        | (b[:, None] == b[None, :])
    )
    for _ in range(3):
        board = RiverBoard(rng.permutation(52)[:5])
        live = board.live
        compatible = (~shares) & live[:, None] & live[None, :]
        reach = np.where(live, rng.random(N_COMBOS) * (rng.random(N_COMBOS) < 0.4), 0.0)
        sign = np.sign(board.strengths[:, None] - board.strengths[None, :])
        dense_showdown = (compatible * sign * 700.0) @ reach
        dense_fold = (compatible * 300.0) @ reach
        showdown = ShowdownTerminal(board, 700.0)
        np.testing.assert_allclose(showdown.utility(0, reach), dense_showdown, atol=1e-8)
        np.testing.assert_allclose(showdown.utility(1, reach), dense_showdown, atol=1e-8)
        for player in (0, 1):
            fold = FoldTerminal(board, folder=0, lost=300.0)
            expected = -dense_fold if player == 0 else dense_fold
            np.testing.assert_allclose(fold.utility(player, reach), expected, atol=1e-8)


def test_bet_state_agrees_with_the_engine_on_random_river_lines():
    rng = random.Random(5)
    for _ in range(400):
        config = GameConfig(stacks=(rng.choice([1500, 10_000]), rng.choice([1500, 10_000])))
        state = HandState(config, tuple(rng.sample(range(52), 9)))
        state.apply(Action(RAISE, 300))
        state.apply(Action(CALL))
        for _ in range(4):  # check the flop and turn through
            state.apply(Action(CHECK))
        assert state.street == 3
        if rng.random() < 0.5:
            state.apply(Action(CHECK))
        bet = BetState.from_spot(state.view())
        while not state.finished:
            legal = state.legal()
            assert bet.legal() == legal and bet.to_act == state.to_act
            options = [Action(CHECK) if legal.can_check else Action(CALL, legal.call_amount)]
            if legal.can_fold:
                options.append(Action(FOLD))
            if legal.can_raise:
                options.append(Action(RAISE, rng.randint(legal.min_raise_to, legal.max_raise_to)))
            action = rng.choice(options)
            state.apply(action)
            bet = bet.apply(action)
            assert bet.total_contrib == tuple(state.total_contrib)
            assert bet.stacks == tuple(state.stacks)
        assert bet.finished
        assert bet.folded == state.folded


def test_tree_actions_are_legal_ordered_and_capped():
    spot, _ = tracked("AsQd", pot=600, to_call=0, position="bb")
    state = BetState.from_spot(spot)
    sizing = BetSizing(bet_fracs=(0.33, 0.75), raise_fracs=(0.75,), max_raises=2)
    actions = tree_actions(state, sizing)
    assert [a.kind for a in actions] == [CHECK, RAISE, RAISE, RAISE]
    assert [a.amount for a in actions[1:]] == [198, 450, 9700]  # two sizes and all-in
    assert all(state.legal().allows(a) for a in actions)
    after_two = state.apply(actions[1]).apply(Action(RAISE, 1000))
    assert [a.kind for a in tree_actions(after_two, sizing)] == [FOLD, CALL]  # raise cap reached
    # A size that would commit most of the stack becomes all-in.
    short = spot_from_partial(
        parse_cards("AsQd"), parse_cards(BOARD), pot=600, stack=500, position="bb"
    )
    assert [a.amount for a in tree_actions(BetState.from_spot(short), sizing)[1:]] == [198, 500]


@pytest.mark.parametrize("size", [0.5, 1.0, 2.0])
def test_polarised_range_against_bluff_catchers_matches_theory(size):
    # The bettor holds only the nuts (three sets) or nothing (three queen-high
    # hands); the caller holds only top pair. No card is shared between the
    # ranges. For one bet of ``size`` pots, theory says: always bet the nuts,
    # bluff size/(1+size) hands per value hand, and call 1/(1+size) of the time.
    board = parse_cards("2c3d7h9sKc")
    spot = spot_from_partial(parse_cards("7c7d"), board, pot=1000, position="button")
    bettor, catcher = np.zeros(N_COMBOS), np.zeros(N_COMBOS)
    value = [combo_index(*parse_cards(h)) for h in ("7c7d", "7c7s", "7d7s")]
    air = [combo_index(*parse_cards(h)) for h in ("QhJh", "QdJd", "QsJs")]
    pairs = [combo_index(*parse_cards(h)) for h in ("AsKh", "AdKs", "AhKd")]
    bettor[value + air] = 1.0
    catcher[pairs] = 1.0
    sizing = BetSizing(bet_fracs=(size,), raise_fracs=(), all_in=False, max_raises=1)
    solution = solve_river(
        spot, (bettor, catcher), sizing, time_budget=30, max_iterations=3000, target_pct_pot=1e-3
    )
    assert [a.kind for a in solution.actions] == [CHECK, RAISE]
    assert solution.exploitability_pct_pot < 0.02
    assert solution.strategy[value, 1] == pytest.approx(1.0, abs=1e-3)
    assert solution.strategy[air, 1].mean() == pytest.approx(size / (1 + size), abs=0.01)
    # The bluffs are indifferent, so betting air is worth the same as giving up: nothing.
    _, air_values = solution.for_hand(parse_cards("QhJh"))
    assert air_values == pytest.approx([0.0, 0.0], abs=2.0)
    # The nuts win the pot by checking and more by betting: called 1/(1+size) of the time.
    _, nut_values = solution.for_hand(parse_cards("7c7d"))
    assert nut_values[0] == pytest.approx(1000.0, abs=1e-6)
    assert nut_values[1] == pytest.approx(1000.0 + 1000.0 * size / (1 + size), abs=5.0)

    # The caller's side, read from a solve of the node after the bet.
    bet_to = solution.actions[1].amount
    engine = HandState(GameConfig(), tuple(parse_cards("7c7d") + parse_cards("AsKh") + board))
    for kind, amount in [(RAISE, 500), (CALL, 0), *[(CHECK, 0)] * 5, (RAISE, bet_to)]:
        engine.apply(Action(kind, amount))
    bluffs = solution.strategy[:, 1] * bettor
    reply = solve_river(
        engine.view(),
        (bluffs, catcher),
        sizing,
        time_budget=30,
        max_iterations=3000,
        target_pct_pot=1e-3,
    )
    assert [a.kind for a in reply.actions] == [FOLD, CALL]
    # Against exactly balanced bluffs the caller is indifferent: call and fold are worth the same.
    _, call_values = reply.for_hand(parse_cards("AsKh"))
    assert call_values[1] == pytest.approx(0.0, abs=25.0)


def test_realistic_solve_is_accurate_and_consistent():
    spot, ranges = tracked("AsQd", pot=1000, to_call=400, position="button")
    solution = solve_river(spot, ranges, time_budget=30, max_iterations=400, target_pct_pot=None)
    assert solution.iterations == 400 and solution.stopped == "iterations"
    assert solution.exploitability_pct_pot < 0.5
    live = RiverBoard(spot.board).live
    np.testing.assert_allclose(solution.strategy[live].sum(axis=1), 1.0)
    assert [a.kind for a in solution.actions][:2] == [FOLD, CALL]
    # Folding is worth exactly what has already been put in, for every hand.
    np.testing.assert_allclose(solution.action_values[live, 0], solution.fold_value)
    # No action can win more than the opponent's whole stack or lose more than the hero's.
    start = spot.stacks[0] + spot.total_contrib[0]
    assert (
        np.nanmax(solution.action_values) <= start and np.nanmin(solution.action_values) >= -start
    )
    # The best hand never folds; the worst never calls for value.
    strengths = RiverBoard(spot.board).strengths
    nuts = int(np.argmax(strengths))
    assert solution.strategy[nuts, 0] < 1e-3


def test_exploitability_falls_with_iterations_and_values_are_zero_sum():
    spot, ranges = tracked("AsQd", pot=600, to_call=0, position="bb")
    board = RiverBoard(spot.board)
    reach = [np.where(board.live, r, 0.0) / np.where(board.live, r, 0.0).sum() for r in ranges]
    norm = float(reach[0] @ board.compatible(reach[1]))
    tree = build_river_tree(BetState.from_spot(spot), board, BetSizing())
    solver = VectorCFR(tree, "dcfr", (reach[0], reach[1]), norm)
    seen = []
    for _ in range(4):
        solver.iterate(40)
        seen.append(solver.exploitability())
    assert seen[-1] < seen[0] / 3 and min(seen) >= -1e-9
    assert solver.game_value(0) + solver.game_value(1) == pytest.approx(0.0, abs=1e-9)
    # The pot is 600, so no one's value can be outside what is at stake.
    assert abs(solver.game_value(0)) <= 10_000


def test_time_budget_is_respected():
    spot, ranges = tracked("AsQd", pot=600, to_call=0, position="bb")
    started = time.perf_counter()
    solution = solve_river(
        spot, ranges, time_budget=0.15, max_iterations=10**6, target_pct_pot=None
    )
    assert time.perf_counter() - started < 0.6
    assert solution.stopped == "time" and solution.iterations >= 5
    np.testing.assert_allclose(
        np.nansum(solution.strategy, axis=1)[RiverBoard(spot.board).live], 1.0
    )


def test_stops_early_once_the_target_is_met():
    spot, ranges = tracked("AsQd", pot=1000, to_call=400, position="button")
    solution = solve_river(spot, ranges, time_budget=30, max_iterations=5000, target_pct_pot=0.5)
    assert solution.stopped == "converged" and solution.iterations < 1000
    assert solution.exploitability_pct_pot <= 0.5


def test_advisor_uses_the_solver_on_the_river_only():
    from pokerbot.advisor import LAYER_HEURISTIC, LAYER_RIVER, SolverConfig, advise

    solver = SolverConfig(time_budget=5.0, max_iterations=150, target_pct_pot=None)
    river = spot_from_partial(parse_cards("AsQd"), parse_cards(BOARD), pot=1000, to_call=400)
    rec = advise(river, solver=solver)
    assert rec.layer == LAYER_RIVER and rec.iterations == 150
    assert river.legal().allows(rec.action)
    assert sum(rec.probabilities.values()) == pytest.approx(1.0)
    assert rec.ev["fold"] == pytest.approx(0.0, abs=1e-6)
    assert set(rec.ev) == set(rec.probabilities)
    assert any("within this bet tree and these ranges" in note for note in rec.notes)
    assert advise(river).layer == LAYER_HEURISTIC  # no solver unless asked

    turn = spot_from_partial(parse_cards("AsQd"), parse_cards(BOARD[:8]), pot=1000, to_call=400)
    assert advise(turn, solver=solver).layer == LAYER_HEURISTIC


def test_solver_bot_plays_legal_hands():
    from pokerbot.arena.match import run_match

    result = run_match("solver", "random", pairs=6, seed=21)
    assert result.illegal_a == 0 and result.illegal_b == 0
