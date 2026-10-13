import numpy as np
import pytest

from pokerbot.cards import CLASS_SIZE, class_from_label
from pokerbot.policy.pushfold import class_deal_probability, solve_pushfold


def test_deal_probability_accounts_for_shared_cards():
    p = class_deal_probability()
    assert p.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(p, p.T)
    # Marginals are the class sizes: 6 combos for a pair, 4 suited, 12 offsuit.
    np.testing.assert_allclose(p.sum(axis=1), CLASS_SIZE / 1326)
    aa, kk = class_from_label("AA"), class_from_label("KK")
    # Two players holding aces need all four aces: 6 ordered pairs of disjoint combos.
    assert p[aa, aa] == pytest.approx(6 / (1326 * 1225))
    assert p[aa, kk] == pytest.approx(36 / (1326 * 1225))


@pytest.mark.parametrize("stack", [5.0, 10.0, 20.0])
def test_pushfold_is_solved_to_high_accuracy(stack):
    solution = solve_pushfold(stack)
    assert solution.exploitability < 1e-5
    aa, worst = class_from_label("AA"), class_from_label("32o")
    assert solution.jam[aa] == pytest.approx(1.0) and solution.call[aa] == pytest.approx(1.0)
    assert solution.call[worst] == pytest.approx(0.0, abs=1e-6)


def test_ranges_tighten_as_stacks_deepen():
    jams = [solve_pushfold(s).jam_frequency for s in (5.0, 10.0, 20.0, 50.0)]
    calls = [solve_pushfold(s).call_frequency for s in (5.0, 10.0, 20.0, 50.0)]
    assert jams == sorted(jams, reverse=True)
    assert calls == sorted(calls, reverse=True)


def test_ten_big_blind_ranges_match_the_published_chart():
    # Widely published values for the 10 big blind push-or-fold equilibrium
    # are about 58% jamming and 37% calling.
    solution = solve_pushfold(10.0)
    assert solution.jam_frequency == pytest.approx(0.583, abs=0.01)
    assert solution.call_frequency == pytest.approx(0.374, abs=0.01)
    assert "AA#" in solution.chart("jam")


def test_a_very_short_stack_calls_everything():
    # With 2 big blinds the big blind already has half its stack in the pot.
    assert solve_pushfold(2.0).call_frequency == pytest.approx(1.0, abs=1e-6)


def test_short_stacked_button_plays_the_pushfold_equilibrium():
    from pokerbot.advisor import advise
    from pokerbot.cards import COMBO_CLASS, parse_cards
    from pokerbot.game.state import GameConfig, HandState
    from pokerbot.policy.base import B_RAISE
    from pokerbot.ranges import RangeTracker

    config = GameConfig(stacks=(800, 800))  # 8 big blinds each
    state = HandState(config, tuple(parse_cards("AsKd" + "7c2d" + "2c3c4c5c9d")))
    spot = state.view()
    strategy = RangeTracker().strategy(spot)
    solution = solve_pushfold(8.0)
    assert strategy.raise_to == 800
    np.testing.assert_allclose(strategy.probs[:, B_RAISE], solution.jam[COMBO_CLASS])
    rec = advise(spot)
    assert rec.action.kind == "raise" and rec.action.amount == 800
    assert any("push-or-fold" in note for note in rec.notes)

    # The big blind then decides by pot odds against that jamming range,
    # which is its best response and so its side of the same equilibrium.
    state.apply(rec.action)
    reply = RangeTracker()
    reply.catch_up(state.view())
    call_frequency = reply.strategy(state.view()).probs[:, 1].mean()
    assert call_frequency == pytest.approx(solution.call_frequency, abs=0.05)

    # Deep stacks do not use the table.
    deep = HandState(GameConfig(), tuple(parse_cards("AsKd" + "7c2d" + "2c3c4c5c9d"))).view()
    assert RangeTracker().strategy(deep).raise_to == 250
