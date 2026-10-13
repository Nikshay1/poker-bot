"""Offline checks of the Slumbot client. No network is used here."""

import pytest

from pokerbot.arena.slumbot import SlumbotError, parse_actions, spot_from_response, to_incr
from pokerbot.cards import parse_cards
from pokerbot.game.state import CALL, CHECK, FOLD, RAISE, Action


def response(action, client_pos, hole="AsKd", board=()):
    return {
        "action": action,
        "client_pos": client_pos,
        "hole_cards": hole_list(hole),
        "board": list(board),
    }


def hole_list(text):
    return [text[i : i + 2] for i in range(0, len(text), 2)]


def test_parse_actions():
    assert parse_actions("") == []
    assert parse_actions("b200c/kk/kb300") == [
        (RAISE, 200),
        (CALL, 0),
        (CHECK, 0),
        (CHECK, 0),
        (CHECK, 0),
        (RAISE, 300),
    ]
    assert parse_actions("b200f") == [(RAISE, 200), (FOLD, 0)]
    with pytest.raises(SlumbotError):
        parse_actions("b200x")


def test_actions_round_trip():
    assert to_incr(Action(FOLD)) == "f"
    assert to_incr(Action(CHECK)) == "k"
    assert to_incr(Action(CALL, 150)) == "c"
    assert to_incr(Action(RAISE, 900)) == "b900"


def test_client_position_zero_is_the_big_blind():
    # The response recorded from the live API: Slumbot, on the button, raised to 200.
    spot = spot_from_response(response("b200", 0, "4d3c"))
    assert spot.seat == 1 and spot.hole == tuple(parse_cards("4d3c"))
    assert spot.to_call == 100 and spot.pot == 300
    assert spot.stacks == (19_800, 19_900)
    assert spot.history == ((0, 0, RAISE, 200),)


def test_client_on_the_button_acts_first():
    spot = spot_from_response(response("", 1))
    assert spot.seat == 0 and spot.to_call == 50 and spot.history == ()


def test_postflop_spot_is_rebuilt_with_the_board():
    # Slumbot raised on the button, we called; on the flop we checked and it bet.
    data = response("b200c/kb300", 0, "AsKd", ["7h", "8h", "2c"])
    spot = spot_from_response(data)
    assert spot.seat == 1 and spot.street == 1
    assert spot.board == tuple(parse_cards("7h8h2c"))
    assert spot.pot == 700 and spot.to_call == 300
    spot.validate()


def test_no_decision_when_the_hand_is_over_or_not_our_turn():
    assert spot_from_response(response("b200f", 0)) is None
    assert spot_from_response(response("b20000c", 1)) is None  # all-in and called
    with pytest.raises(ValueError):
        spot_from_response(response("b150", 0))  # below the minimum raise
