import numpy as np
import pytest

from pokerbot.solver.toy_games import KUHN_VALUE, kuhn_tree, leduc_tree
from pokerbot.solver.vector_cfr import VARIANTS, VectorCFR, regret_matching

LEDUC_VALUE = -0.0856  # first player's equilibrium value, as published


def test_regret_matching():
    regret = np.array([[2.0, 1.0, -1.0], [-1.0, -2.0, -3.0], [0.0, 0.0, 4.0]])
    strategy = regret_matching(regret)
    np.testing.assert_allclose(strategy[0], [2 / 3, 1 / 3, 0.0])
    np.testing.assert_allclose(strategy[1], [1 / 3, 1 / 3, 1 / 3])  # no positive regret
    np.testing.assert_allclose(strategy[2], [0.0, 0.0, 1.0])


@pytest.mark.parametrize("variant", VARIANTS)
def test_kuhn_converges_to_the_known_value(variant):
    solver = VectorCFR(kuhn_tree(), variant)
    solver.iterate(2000)
    assert solver.game_value(0) == pytest.approx(KUHN_VALUE, abs=2e-4)
    assert solver.game_value(1) == pytest.approx(-KUHN_VALUE, abs=2e-4)
    assert solver.exploitability() < 1e-3
    # Neither player can gain much by deviating.
    assert solver.best_response_value(0) - solver.game_value(0) < 2e-3
    assert solver.best_response_value(1) - solver.game_value(1) < 2e-3


def test_kuhn_equilibrium_structure():
    solver = VectorCFR(kuhn_tree(), "dcfr")
    solver.iterate(5000)
    root = solver.tree.root
    first = solver.average_strategy(root)  # rows J, Q, K; columns check, bet
    after_check = solver.average_strategy(root.children[0])  # second player: check, bet
    after_bet = solver.average_strategy(root.children[1])  # second player: fold, call
    facing_bet = solver.average_strategy(root.children[0].children[1])  # first: fold, call

    # The first player's equilibria form a family indexed by how often it
    # bluffs the Jack: alpha in [0, 1/3]. It then bets the King 3 * alpha of
    # the time and calls with the Queen alpha + 1/3 of the time.
    alpha = first[0, 1]
    assert 0.0 <= alpha <= 1 / 3 + 1e-3
    assert first[1, 1] == pytest.approx(0.0, abs=5e-3)  # never bets the Queen
    assert first[2, 1] == pytest.approx(3 * alpha, abs=1e-2)
    assert facing_bet[1, 1] == pytest.approx(alpha + 1 / 3, abs=1e-2)
    # The second player's equilibrium strategy is unique.
    assert after_check[0, 1] == pytest.approx(1 / 3, abs=1e-2)  # bluffs the Jack a third
    assert after_check[2, 1] == pytest.approx(1.0, abs=5e-3)  # always bets the King
    assert after_bet[1, 1] == pytest.approx(1 / 3, abs=1e-2)  # calls with the Queen a third
    assert after_bet[0, 0] == pytest.approx(1.0, abs=5e-3)  # folds the Jack
    assert after_bet[2, 1] == pytest.approx(1.0, abs=5e-3)  # calls with the King


def test_leduc_converges_to_the_published_value():
    tree = leduc_tree()
    assert len(tree.decisions) == 186 and tree.n_terminals == 274
    solver = VectorCFR(tree, "dcfr")
    solver.iterate(300)
    assert solver.game_value(0) == pytest.approx(LEDUC_VALUE, abs=1e-3)
    assert solver.exploitability() < 5e-3


def test_exploitability_is_non_negative_and_falls():
    solver = VectorCFR(leduc_tree(), "dcfr")
    seen = []
    for _ in range(4):
        solver.iterate(25)
        seen.append(solver.exploitability())
    assert all(x >= -1e-12 for x in seen)
    assert seen[-1] < seen[0] / 5


def test_untrained_strategy_is_uniform_and_exploitable():
    solver = VectorCFR(kuhn_tree(), "cfr")
    np.testing.assert_allclose(solver.average_strategy(solver.tree.root), 0.5)
    assert solver.exploitability() > 0.1


def test_values_are_zero_sum():
    solver = VectorCFR(leduc_tree(), "cfr+")
    solver.iterate(20)
    assert solver.game_value(0) + solver.game_value(1) == pytest.approx(0.0, abs=1e-12)


def test_discounted_cfr_beats_plain_cfr_on_leduc():
    # An observation recorded as a regression check, not a theorem.
    plain, discounted = VectorCFR(leduc_tree(), "cfr"), VectorCFR(leduc_tree(), "dcfr")
    plain.iterate(200)
    discounted.iterate(200)
    assert discounted.exploitability() < plain.exploitability() / 3
