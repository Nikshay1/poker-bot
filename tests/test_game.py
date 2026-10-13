import dataclasses
import random

import pytest

from pokerbot.cards import parse_cards
from pokerbot.game.state import (
    CALL,
    CHECK,
    FOLD,
    RAISE,
    Action,
    GameConfig,
    HandState,
    Spot,
    SpotError,
)

CFG = GameConfig()  # 50/100 blinds, 10,000 each


def deal(seat0: str, seat1: str, board: str) -> tuple[int, ...]:
    return tuple(parse_cards(seat0 + seat1 + board))


def new_hand(seat0="AsAd", seat1="KsKd", board="2c7h9dJcQs", config=CFG) -> HandState:
    return HandState(config, deal(seat0, seat1, board))


def play(state: HandState, *actions) -> HandState:
    for kind, *amount in actions:
        state.apply(Action(kind, amount[0] if amount else 0))
    return state


def test_blinds_and_first_action():
    s = new_hand()
    assert s.to_act == 0 and s.pot == 150
    assert s.stacks == [9950, 9900]
    legal = s.legal()
    assert legal.can_fold and not legal.can_check
    assert legal.call_amount == 50
    assert (legal.min_raise_to, legal.max_raise_to) == (200, 10_000)


def test_limp_gives_big_blind_the_option():
    s = play(new_hand(), (CALL,))
    assert s.street == 0 and s.to_act == 1
    legal = s.legal()
    assert legal.can_check and not legal.can_fold
    assert legal.min_raise_to == 200
    play(s, (CHECK,))
    assert s.street == 1 and len(s.board) == 3 and s.to_act == 1  # big blind acts first postflop


def test_minimum_raise_follows_the_last_raise_size():
    s = play(new_hand(), (RAISE, 300))  # a raise of 200 over the big blind
    assert s.legal().min_raise_to == 500
    play(s, (RAISE, 900))  # a raise of 600
    assert s.legal().min_raise_to == 1500
    with pytest.raises(ValueError):
        s.apply(Action(RAISE, 1400))


def test_minimum_bet_after_the_flop_is_one_big_blind():
    s = play(new_hand(), (CALL,), (CHECK,))
    legal = s.legal()
    assert (legal.min_raise_to, legal.max_raise_to) == (100, 9900)
    play(s, (RAISE, 100))
    assert s.legal().min_raise_to == 200


def test_fold_pays_what_the_folder_put_in():
    s = play(new_hand(), (RAISE, 300), (RAISE, 900), (FOLD,))
    assert s.finished and s.folded == 0
    assert s.payoffs == (-300, 300)
    assert s.board == []


def test_showdown_winner_takes_the_pot():
    s = play(
        new_hand(),
        (RAISE, 300),
        (CALL,),
        (CHECK,),
        (RAISE, 400),
        (CALL,),
        (CHECK,),
        (CHECK,),
        (CHECK,),
        (CHECK,),
    )
    assert s.finished and s.folded is None
    assert s.payoffs == (700, -700)  # aces beat kings
    assert len(s.board) == 5


def test_split_pot_pays_nothing():
    s = new_hand("2c3d", "4c5d", "AhKdQcJsTh")  # the board plays
    play(s, (RAISE, 300), (CALL,), *([(CHECK,)] * 6))
    assert s.finished and s.payoffs == (0, 0)


def test_all_in_and_call_runs_out_the_board():
    s = play(new_hand(), (RAISE, 10_000), (CALL,))
    assert s.finished and len(s.board) == 5
    assert s.payoffs == (10_000, -10_000)
    assert s.stacks == [0, 0]


def test_only_call_or_fold_against_an_all_in():
    s = play(new_hand(), (RAISE, 300), (RAISE, 10_000))
    legal = s.legal()
    assert legal.can_fold and legal.call_amount == 9700 and not legal.can_raise


def test_raise_is_capped_at_the_shorter_stack():
    config = GameConfig(stacks=(10_000, 2_000))
    s = new_hand(config=config)
    assert s.legal().max_raise_to == 2_000
    play(s, (RAISE, 2_000), (CALL,))
    assert s.payoffs == (2_000, -2_000)
    assert s.stacks == [8_000, 0]


def test_short_all_in_below_the_minimum_raise_is_allowed():
    config = GameConfig(stacks=(10_000, 450))
    s = play(new_hand(config=config), (RAISE, 300))
    legal = s.legal()  # a full raise would be to 500, but the big blind only has 450
    assert (legal.min_raise_to, legal.max_raise_to) == (450, 450)
    play(s, (RAISE, 450))
    assert s.legal().call_amount == 150 and not s.legal().can_raise


def test_illegal_actions_are_rejected():
    s = new_hand()
    with pytest.raises(ValueError):
        s.apply(Action(CHECK))  # facing the big blind
    play(s, (CALL,))
    with pytest.raises(ValueError):
        s.apply(Action(FOLD))  # nothing to fold to
    with pytest.raises(ValueError):
        s.apply(Action(RAISE, 150))  # below the minimum


def test_view_hides_the_opponent_and_the_future_board():
    s = play(new_hand(), (RAISE, 300), (CALL,))
    spot = s.view()
    assert spot.seat == 1 and spot.hole == tuple(parse_cards("KsKd"))
    assert spot.board == tuple(parse_cards("2c7h9d"))
    assert spot.pot == 600 and spot.to_call == 0
    visible = repr(dataclasses.asdict(spot))
    hidden = parse_cards("AsAd") + parse_cards("JcQs")  # opponent's hand, turn and river
    fields = [v for v in dataclasses.asdict(spot).values()]
    flat = set()
    for value in fields:
        if isinstance(value, tuple):
            flat.update(x for x in value if isinstance(x, int))
    assert not flat & set(hidden), visible
    assert not hasattr(spot, "deck") and not hasattr(spot, "_hole")


def test_random_playouts_conserve_chips_and_stay_legal():
    rng = random.Random(42)
    for _ in range(3000):
        config = GameConfig(
            stacks=(rng.choice([300, 2000, 10_000]), rng.choice([300, 2000, 10_000]))
        )
        s = HandState(config, tuple(rng.sample(range(52), 9)))
        steps = 0
        while not s.finished:
            legal = s.legal()
            options = []
            if legal.can_fold:
                options.append(Action(FOLD))
            if legal.can_check:
                options.append(Action(CHECK))
            if legal.call_amount:
                options.append(Action(CALL))
            if legal.can_raise:
                options.append(Action(RAISE, rng.randint(legal.min_raise_to, legal.max_raise_to)))
                options.append(Action(RAISE, legal.min_raise_to))
            action = rng.choice(options)
            assert legal.allows(action)
            assert s.view().legal() == legal
            s.apply(action)
            # Chips are conserved at every step.
            assert sum(s.stacks) + s.pot == sum(config.stacks)
            assert min(s.stacks) >= 0
            steps += 1
            assert steps < 200
        assert sum(s.payoffs) == 0
        assert abs(s.payoffs[0]) <= min(config.stacks)
        if s.folded is None:
            assert s.total_contrib[0] == s.total_contrib[1]


def test_spot_validation_reports_structured_errors():
    good = new_hand().view()
    good.validate()
    cases = {
        "duplicate_card": dataclasses.replace(good, board=(good.hole[0], 1, 2)),
        "bad_board": dataclasses.replace(good, board=(1, 2)),
        "bad_seat": dataclasses.replace(good, seat=2),
        "negative_chips": dataclasses.replace(good, stacks=(-1, 100)),
        "not_to_act": dataclasses.replace(
            good, street_contrib=(300, 100), total_contrib=(300, 100)
        ),
        "no_decision": dataclasses.replace(good, stacks=(0, 100)),
    }
    for code, spot in cases.items():
        with pytest.raises(SpotError) as err:
            spot.validate()
        assert err.value.code == code
        assert err.value.as_dict()["error"] == code


def test_spot_is_immutable():
    spot: Spot = new_hand().view()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spot.seat = 1
