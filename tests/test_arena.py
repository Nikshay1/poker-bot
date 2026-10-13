import numpy as np
import pytest

from pokerbot.arena.match import play_hand, run_match
from pokerbot.bots.basic import BOTS, Bot, make_bot
from pokerbot.game.state import Action, GameConfig, Spot

CFG = GameConfig()


def test_identical_deterministic_bots_break_even_exactly():
    result = run_match("call", "call", pairs=200, seed=1)
    # Same cards, seats swapped, same play: every pair cancels.
    assert result.bb_per_100 == 0.0 and result.stdev_pair_bb == 0.0
    assert result.hands == 400 and result.illegal_a == result.illegal_b == 0


def test_match_is_reproducible_and_seed_dependent():
    a = run_match("random", "call", pairs=150, seed=3)
    b = run_match("random", "call", pairs=150, seed=3)
    c = run_match("random", "call", pairs=150, seed=4)
    assert a.bb_per_100 == b.bb_per_100 and a.ci95 == b.ci95
    assert a.bb_per_100 != c.bb_per_100


def test_swapping_the_bots_negates_the_result():
    a = run_match("equity", "call", pairs=6, seed=5)
    b = run_match("call", "equity", pairs=6, seed=5)
    assert a.bb_per_100 == pytest.approx(-b.bb_per_100)


def test_confidence_interval_is_computed_over_pairs():
    result = run_match("random", "call", pairs=300, seed=7)
    assert result.ci95 == pytest.approx(1.96 * result.stdev_pair_bb / np.sqrt(300) * 100)
    # Duplicate dealing removes card luck, so a pair's mean varies less than
    # two independent hands would.
    assert result.stdev_pair_bb < result.stdev_hand_bb / np.sqrt(2)


def test_parallel_workers_give_the_same_result():
    serial = run_match("random", "call", pairs=64, seed=9, workers=1)
    parallel = run_match("random", "call", pairs=64, seed=9, workers=2)
    assert serial.bb_per_100 == parallel.bb_per_100


def test_bots_only_receive_their_own_view():
    seen = []

    class Spy(Bot):
        def act(self, spot):
            seen.append(spot)
            legal = spot.legal()
            return Action("check") if legal.can_check else Action("call", legal.call_amount)

    deck = tuple(range(9))
    play_hand(Spy(), Spy(), CFG, deck)
    assert seen and all(type(s) is Spot for s in seen)
    for spot in seen:
        assert spot.hole == (deck[0], deck[1]) if spot.seat == 0 else (deck[2], deck[3])
        assert set(spot.board) <= set(deck[4 : 4 + len(spot.board)])


def test_illegal_actions_are_counted_and_replaced():
    class Cheat(Bot):
        def act(self, spot):
            return Action("raise", 10**9)

    payoffs, stats = play_hand(Cheat(), make_bot("call"), CFG, tuple(range(9)))
    assert stats["illegal"][0] > 0 and stats["illegal"][1] == 0
    assert sum(payoffs) == 0


@pytest.mark.parametrize("name", sorted(BOTS))
def test_every_bot_plays_legal_hands_against_random(name):
    pairs = {"heuristic": 8, "equity": 8, "solver": 4}.get(name, 100)
    result = run_match(name, "random", pairs=pairs, seed=11)
    assert result.illegal_a == 0 and result.illegal_b == 0
