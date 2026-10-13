import itertools
import random

import numpy as np
import pytest

from pokerbot.cards import COMBO_CARDS, N_COMBOS, combo_index, parse_cards
from pokerbot.equity import (
    hand_vs_hand,
    monte_carlo_equity,
    range_equity,
    range_sums,
    uniform_range,
)
from pokerbot.evaluator import evaluate


def naive_range_equity(board, weights):
    """Direct all-pairs computation, independent of the sorted method."""
    board = list(board)
    remaining = [c for c in range(52) if c not in board]
    combos = [tuple(pair) for pair in COMBO_CARDS.tolist()]
    win = np.zeros(N_COMBOS)
    tie = np.zeros(N_COMBOS)
    total = np.zeros(N_COMBOS)
    for runout in itertools.combinations(remaining, 5 - len(board)):
        full = board + list(runout)
        dead = set(full)
        live = [k for k in range(N_COMBOS) if not (set(combos[k]) & dead)]
        strength = {
            k: int(evaluate(np.array(full + list(combos[k]), dtype=np.int64))) for k in live
        }
        weighted = [(o, float(weights[o])) for o in live if weights[o] != 0.0]
        for h in live:
            ha, hb = combos[h]
            for o, w in weighted:
                oa, ob = combos[o]
                if oa in (ha, hb) or ob in (ha, hb):
                    continue
                total[h] += w
                if strength[h] > strength[o]:
                    win[h] += w
                elif strength[h] == strength[o]:
                    tie[h] += w
    return win, tie, total


def sparse_weights(rng, n_live=60):
    weights = np.zeros(N_COMBOS)
    for k in rng.sample(range(N_COMBOS), n_live):
        weights[k] = rng.random()
    return weights


def test_river_range_sums_match_all_pairs():
    rng = random.Random(1)
    for _ in range(3):
        board = rng.sample(range(52), 5)
        weights = sparse_weights(rng, 200)
        fast = range_sums(board, weights)
        slow = naive_range_equity(board, weights)
        for f, s in zip(fast, slow):
            np.testing.assert_allclose(f, s, atol=1e-9)


def test_turn_range_sums_match_all_pairs():
    rng = random.Random(2)
    board = rng.sample(range(52), 4)
    weights = sparse_weights(rng, 40)
    fast = range_sums(board, weights)
    slow = naive_range_equity(board, weights)
    for f, s in zip(fast, slow):
        np.testing.assert_allclose(f, s, atol=1e-9)


def test_flop_equity_matches_hand_vs_hand():
    # Against a single opponent combo, range equity must equal exact hand-vs-hand equity.
    board = parse_cards("7h8h2c")
    villain = parse_cards("AhKh")
    weights = np.zeros(N_COMBOS)
    weights[combo_index(*villain)] = 1.0
    eq = range_equity(board, weights)
    for text in ["9s9d", "7s7d", "AsAd", "JhTh", "2d3d"]:
        hero = parse_cards(text)
        win, tie, _ = hand_vs_hand(hero, villain, board)
        assert eq[combo_index(*hero)] == pytest.approx(win + tie / 2, abs=1e-12)
    # Combos sharing a card with the opponent's only hand face nothing.
    assert np.isnan(eq[combo_index(*parse_cards("AhQd"))])


def test_hand_vs_hand_is_complementary():
    a, b = parse_cards("AsAh"), parse_cards("KsKh")
    win, tie, loss = hand_vs_hand(a, b)
    win2, tie2, loss2 = hand_vs_hand(b, a)
    assert win + tie + loss == pytest.approx(1.0)
    assert (win, tie, loss) == pytest.approx((loss2, tie2, win2))
    # Aces against kings sharing both suits; 1,712,304 boards enumerated.
    assert 0.81 < win + tie / 2 < 0.83


def test_hand_vs_hand_known_river_cases():
    board = parse_cards("AhKdQcJsTh")  # broadway on board: always a tie
    assert hand_vs_hand(parse_cards("2c3d"), parse_cards("4c5d"), board) == (0.0, 1.0, 0.0)
    board = parse_cards("AhKd7c7s2h")
    assert hand_vs_hand(parse_cards("AsAd"), parse_cards("KsKc"), board) == (1.0, 0.0, 0.0)


def test_uniform_range_equities_average_to_half():
    # Against a uniform range the game is symmetric, so the mean over all
    # live combos, weighted by how many opponents each faces, is exactly 0.5.
    board = parse_cards("7h8h2cKd")
    win, tie, total = range_sums(board, uniform_range())
    assert (win.sum() + 0.5 * tie.sum()) / total.sum() == pytest.approx(0.5, abs=1e-12)
    eq = range_equity(board, uniform_range())
    assert np.isnan(eq).sum() == 1326 - 48 * 47 // 2
    assert np.nanmin(eq) >= 0.0 and np.nanmax(eq) <= 1.0


def test_monte_carlo_agrees_with_exact_within_error():
    board = parse_cards("7h8h2c")
    hero = parse_cards("AsAd")
    exact = range_equity(board, uniform_range())[combo_index(*hero)]
    mc = monte_carlo_equity(hero, board, [uniform_range()], n_samples=40_000, seed=5)
    assert abs(mc["equity"] - exact) < 5 * mc["stderr"]
    assert mc["samples"] == 40_000


def test_monte_carlo_multiway_lowers_equity():
    hero = parse_cards("AsAd")
    one = monte_carlo_equity(hero, [], [uniform_range()], n_samples=20_000, seed=1)
    three = monte_carlo_equity(hero, [], [uniform_range()] * 3, n_samples=20_000, seed=1)
    assert 0.83 < one["equity"] < 0.87  # aces against one random hand
    assert three["equity"] < one["equity"] - 0.15


def test_monte_carlo_rejects_impossible_range():
    hero = parse_cards("AsAd")
    weights = np.zeros(N_COMBOS)
    weights[combo_index(*parse_cards("AsKd"))] = 1.0  # blocked by the hero's ace
    with pytest.raises(ValueError):
        monte_carlo_equity(hero, [], [weights])


def test_radix_sort_orders_live_combos_by_strength():
    from pokerbot.equity import sort_by_strength
    from pokerbot.evaluator import combo_strengths

    for text in ("7h8h2cKd3s", "AhKhQhJhTh", "2c2d2h2sKd"):
        strengths = combo_strengths(parse_cards(text))
        order = sort_by_strength(strengths)
        assert len(order) == (strengths >= 0).sum() == 47 * 46 // 2
        assert np.all(np.diff(strengths[order]) >= 0)
        expected = np.argsort(strengths, kind="stable")[-len(order) :]
        np.testing.assert_array_equal(order, expected)  # stable, like a stable argsort
