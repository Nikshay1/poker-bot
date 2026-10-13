import numpy as np
import pytest

from pokerbot.advisor import advise, spot_from_partial
from pokerbot.cards import COMBO_CLASS, N_COMBOS, class_from_label, combo_index, parse_cards
from pokerbot.game.state import CALL, CHECK, FOLD, RAISE, Action, GameConfig, HandState, SpotError
from pokerbot.policy.base import (
    B_CALL,
    B_FOLD,
    B_RAISE,
    PolicyConfig,
    last_bet_risk,
    take_by_order,
)
from pokerbot.policy.multiway import advise_multiway, top_fraction_range
from pokerbot.ranges import RangeTracker, replay

CFG = PolicyConfig()


def partial(hole, board="", **kw):
    return spot_from_partial(parse_cards(hole), parse_cards(board), **kw)


def combos_of(label):
    return np.flatnonzero(COMBO_CLASS == class_from_label(label))


def test_take_by_order_spends_exactly_the_budget():
    weights = np.array([1.0, 2.0, 0.0, 3.0])
    order = np.array([3, 0, 1, 2])
    taken = take_by_order(weights, order, 3.5)
    assert taken.tolist() == [0.5, 0.0, 0.0, 1.0]
    assert (taken * weights).sum() == pytest.approx(3.5)
    assert take_by_order(weights, order, 0.0).sum() == 0.0
    assert take_by_order(weights, order, 100.0).tolist() == [1.0, 1.0, 0.0, 1.0]


def test_button_opening_range():
    spot = partial("AsKd")
    strategy = RangeTracker(CFG).strategy(spot)
    np.testing.assert_allclose(strategy.probs.sum(axis=1), 1.0)
    assert strategy.raise_to == 250
    assert strategy.probs[:, B_RAISE].mean() == pytest.approx(CFG.open_frac, abs=0.005)
    assert strategy.probs[combos_of("AA"), B_RAISE].min() == 1.0
    assert strategy.probs[combos_of("72o"), B_FOLD].min() == 1.0


def test_big_blind_defends_a_sensible_share_against_an_open():
    spot = partial("QsQd", position="bb", to_call=150, pot=350)
    tracker = RangeTracker(CFG)
    tracker.catch_up(spot)
    strategy = tracker.strategy(spot)
    np.testing.assert_allclose(strategy.probs.sum(axis=1), 1.0)
    reraise = strategy.probs[:, B_RAISE].mean()
    defend = 1.0 - strategy.probs[:, B_FOLD].mean()
    assert 0.08 < reraise < 0.35
    assert 0.5 < defend < 0.9
    assert strategy.probs[combos_of("AA"), B_RAISE].min() == 1.0
    assert strategy.probs[combos_of("72o"), B_FOLD].min() == 1.0
    # The button's range has been narrowed to its opening range.
    opened = tracker.ranges[0]
    assert opened[combos_of("AA")].min() > 5 * opened[combos_of("72o")].max()


def test_calling_an_all_in_is_decided_by_pot_odds_alone():
    # Button shoves 100 big blinds; calling ends the betting.
    state = HandState(GameConfig(), tuple(range(9)))
    state.apply(Action(RAISE, 10_000))
    spot = state.view()
    tracker = RangeTracker(CFG)
    tracker.catch_up(spot)
    strategy = tracker.strategy(spot)
    assert strategy.raise_to is None
    need = spot.to_call / (spot.pot + spot.to_call)
    calls = strategy.probs[:, B_CALL] > 0.5
    assert np.all(strategy.equity[calls] >= need)
    assert np.all(strategy.equity[~calls] < need)


def test_postflop_rows_are_distributions_and_the_nuts_never_folds():
    for kw in (dict(pot=600, to_call=0), dict(pot=1000, to_call=400)):
        spot = partial("AsQc", "7h8h2cKd3s", **kw)
        tracker = RangeTracker(CFG)
        tracker.catch_up(spot)
        strategy = tracker.strategy(spot)
        live = ~np.isnan(strategy.equity)
        np.testing.assert_allclose(strategy.probs[live].sum(axis=1), 1.0)
        assert strategy.probs.min() >= 0.0
        nuts = combo_index(*parse_cards("KsKc"))  # top set on this board
        assert strategy.probs[nuts, B_FOLD] == 0.0
        assert strategy.probs[nuts, B_RAISE] == 1.0


def test_minimum_defence_guard_limits_folding():
    spot = partial("9s4d", "7h8h2cKd3s", pot=1200, to_call=600, position="bb")
    tracker = RangeTracker(CFG)
    tracker.catch_up(spot)
    strategy = tracker.strategy(spot)
    weights = np.where(np.isnan(strategy.equity), 0.0, tracker.ranges[spot.seat])
    folded = (weights * strategy.probs[:, B_FOLD]).sum() / weights.sum()
    risk = last_bet_risk(spot)
    assert risk == 600
    defend = 1.0 - risk / spot.pot
    assert folded <= 1.0 - CFG.mdf_guard * defend + 1e-9


def test_river_bluffs_use_the_weakest_hands():
    spot = partial("AsQd", "7h8h2cKd3s", pot=600, to_call=0, position="bb")
    tracker = RangeTracker(CFG)
    tracker.catch_up(spot)
    strategy = tracker.strategy(spot)
    weights = np.where(np.isnan(strategy.equity), 0.0, tracker.ranges[spot.seat])
    bets = strategy.probs[:, B_RAISE]
    value = (bets > 0) & (strategy.equity >= 0.6)
    bluff = (bets > 0) & (strategy.equity < CFG.bluff_ceiling)
    assert (weights * bets * value).sum() > 0 and (weights * bets * bluff).sum() > 0
    checked = (weights > 0) & (bets == 0)
    # Every bluff is weaker than every hand that checks behind.
    assert strategy.equity[bluff].max() <= strategy.equity[checked].min() + 1e-12
    # Bluff weight is the value weight times s / (1 + s) for the bet size used.
    size = (strategy.raise_to - spot.street_contrib[spot.seat]) / spot.pot
    ratio = (weights * bets * bluff).sum() / (weights * bets * value).sum()
    assert ratio == pytest.approx(size / (1 + size), rel=1e-6)


def test_ranges_never_collapse_and_ignore_the_hero_cards():
    spot = partial("AsQd", "7h8h2cKd", pot=1000, to_call=400)
    other = partial("2s2d", "7h8h2cKd", pot=1000, to_call=400)
    a, b = RangeTracker(CFG), RangeTracker(CFG)
    a.catch_up(spot)
    b.catch_up(other)
    for seat in (0, 1):
        assert a.ranges[seat].max() == pytest.approx(1.0)
        live = a.ranges[seat] > 0
        assert live.sum() == 48 * 47 // 2  # every combo off the board keeps some weight
        # Public ranges depend on the board and the betting only.
        np.testing.assert_allclose(a.ranges[seat], b.ranges[seat])


def test_replay_rejects_a_history_that_does_not_fit():
    spot = partial("AsKd", "7h8h2c", pot=1000, to_call=400)
    assert len(replay(spot)) == len(spot.history)
    import dataclasses

    broken = dataclasses.replace(spot, history=spot.history[:-1])
    with pytest.raises(SpotError) as err:
        replay(broken)
    assert err.value.code == "bad_history"
    illegal = dataclasses.replace(spot, history=((0, 0, RAISE, 120),) + spot.history[1:])
    with pytest.raises(SpotError):
        replay(illegal)


def test_partial_input_lists_its_assumptions():
    spot = partial("AsKd", "7h8h2c", pot=1000, to_call=400)
    assert spot.pot == 1000 and spot.to_call == 400 and spot.seat == 0
    assert spot.history == ((0, 0, RAISE, 300), (0, 1, CALL, 200), (1, 1, RAISE, 400))
    text = " ".join(spot.assumptions)
    assert "button raised to 300" in text and "bet 400" in text
    rec = advise(spot)
    assert rec.assumptions == list(spot.assumptions)
    assert spot.legal().allows(rec.action)
    assert rec.ev[CALL] == pytest.approx(rec.equity * 1400 - 400)

    limped = partial("AsKd", "7h8h2c", pot=200, position="bb")
    assert limped.history[:2] == ((0, 0, CALL, 50), (0, 1, CHECK, 0))
    with_stack = partial("AsKd", "7h8h2c", pot=1000, to_call=400, stack=2500)
    assert with_stack.stacks[0] == 2500


@pytest.mark.parametrize(
    "kwargs, code",
    [
        (dict(hole="AsKd", board="7h8h2c"), "missing_pot"),
        (dict(hole="AsKd", board="7h8h2c", pot=1000, to_call=1000), "bad_to_call"),
        (dict(hole="AsKd", board="7h8h2c", pot=250), "bad_pot"),
        (dict(hole="AsKd", board="7h8h", pot=600), "bad_board"),
        (dict(hole="AsKd", board="As8h2c", pot=600), "duplicate_card"),
        (dict(hole="AsKd", board="7h8h2c", pot=600, to_call=20), "bad_partial"),
        (dict(hole="AsKd", position="cutoff"), "bad_position"),
    ],
)
def test_partial_input_errors_are_structured(kwargs, code):
    hole = kwargs.pop("hole")
    board = kwargs.pop("board", "")
    with pytest.raises(SpotError) as err:
        spot_from_partial(parse_cards(hole), parse_cards(board), **kwargs)
    assert err.value.code == code


def test_advise_is_deterministic_without_a_generator():
    spot = partial("9h6h", "7h8h2c", pot=1000, to_call=400)
    first, second = advise(spot), advise(spot)
    assert first.action == second.action and first.probabilities == second.probabilities
    assert first.layer == "heuristic"
    assert sum(first.probabilities.values()) == pytest.approx(1.0)


def test_multiway_equity_analysis():
    hole, board = parse_cards("AsAd"), parse_cards("7h8h2c")
    two = advise_multiway(hole, board, pot=1000, to_call=400, n_opponents=2, seed=1)
    five = advise_multiway(hole, board, pot=1000, to_call=400, n_opponents=5, seed=1)
    assert two.equity > five.equity > five.fair_share
    assert two.ev[CALL] == pytest.approx(two.equity * 1400 - 400)
    assert two.action.kind in (CALL, RAISE)
    weak = advise_multiway(parse_cards("3s2d"), board, pot=1000, to_call=400, n_opponents=3)
    assert weak.action.kind == FOLD
    with pytest.raises(SpotError) as err:
        advise_multiway(hole, board, pot=1000, to_call=400, stack=300, n_opponents=2)
    assert err.value.code == "side_pot_unsupported"
    assert top_fraction_range(0.30).sum() / N_COMBOS == pytest.approx(0.30, abs=0.02)


def test_assumed_checks_are_not_evidence():
    import dataclasses

    # A river spot from partial input: the flop and turn checks are filler.
    spot = partial("AsQd", "7h8h2cKd3s", pot=1000, to_call=400)
    assert spot.unobserved == (2, 3, 4, 5)
    assert [spot.history[i][2] for i in spot.unobserved] == [CHECK] * 4
    assert any("not used to narrow ranges" in a for a in spot.assumptions)

    filler, observed = RangeTracker(CFG), RangeTracker(CFG)
    filler.catch_up(spot)
    observed.catch_up(dataclasses.replace(spot, unobserved=()))  # as if the checks were seen
    kings = combo_index(*parse_cards("KsKc"))  # top set
    air = combo_index(*parse_cards("5c4d"))
    # Seeing two checks makes a set unlikely; assuming them must not.
    assert filler.ranges[1][kings] / filler.ranges[1][air] > 20 * (
        observed.ranges[1][kings] / observed.ranges[1][air]
    )
    # A complete state from the engine has nothing unobserved.
    engine = HandState(GameConfig(), tuple(range(9)))
    assert engine.view().unobserved == ()
