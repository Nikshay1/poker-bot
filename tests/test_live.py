"""Following a real hand as it is entered, and the web server on top of it."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from pokerbot.advisor import SolverConfig
from pokerbot.cards import parse_cards
from pokerbot.game.state import SpotError
from pokerbot.live import BOARD, HERO, OPPONENT, OVER, LiveHand, advise_live, follow
from pokerbot.web.server import make_handler

FAST = SolverConfig(time_budget=5.0, max_iterations=60)


def hand(hole="AsKd", seat=0, board="", actions=(), stacks=(10_000, 10_000)):
    return LiveHand(
        hole=tuple(parse_cards(hole)),
        seat=seat,
        stacks=stacks,
        board=tuple(parse_cards(board)),
        actions=tuple(actions),
    )


def test_button_acts_first_and_gets_advice():
    status = advise_live(hand())
    assert status.waiting_for == HERO
    assert (status.pot, status.legal.call_amount) == (150, 50)
    assert status.spot.hole == tuple(parse_cards("AsKd"))
    assert status.recommendation.action.kind == "raise"
    assert status.to_dict()["recommendation"]["text"] == "raise to 250"


def test_big_blind_waits_for_the_button():
    status = follow(hand(seat=1))
    assert status.waiting_for == OPPONENT
    assert status.spot is None
    status = advise_live(hand(seat=1, actions=[("raise", 250)]))
    assert status.waiting_for == HERO
    assert status.history[0] == {
        "street": "preflop", "who": "opponent", "kind": "raise", "amount": 250,
        "text": "raise to 250",
    }  # fmt: skip


def test_board_is_asked_for_street_by_street():
    line = [("raise", 250), ("call", 0)]
    status = follow(hand(actions=line))
    assert (status.waiting_for, status.cards_needed) == (BOARD, 3)
    assert follow(hand(actions=line, board="7h8h")).cards_needed == 1
    # After the flop the big blind acts first, so the button waits.
    assert follow(hand(actions=line, board="7h8h2c")).waiting_for == OPPONENT
    line += [("check", 0), ("raise", 300), ("call", 0)]
    status = follow(hand(actions=line, board="7h8h2c"))
    assert (status.waiting_for, status.cards_needed, status.pot) == (BOARD, 1, 1100)
    assert [h["text"] for h in status.history[2:]] == ["check", "bet 300", "call 300"]


def test_river_decision_comes_from_the_solver():
    line = [("raise", 250), ("call", 0)] + [("check", 0)] * 4 + [("raise", 400)]
    status = advise_live(hand(hole="AsAd", actions=line, board="9s8s7d6c2h"), FAST)
    assert status.waiting_for == HERO
    reply = status.to_dict()
    advice = reply["recommendation"]
    assert advice["layer"] == "river solver"
    assert reply["history"][-1]["text"] == "bet 400"
    # Every option is a concrete action the page can send straight back.
    for option in advice["options"]:
        again = hand(
            hole="AsAd", actions=[*line, (option["kind"], option["amount"])], board="9s8s7d6c2h"
        )
        assert follow(again).waiting_for in (OPPONENT, OVER)
    assert sum(o["probability"] for o in advice["options"]) == pytest.approx(1.0)


def test_fold_ends_the_hand_with_a_known_result():
    status = follow(hand(seat=1, actions=[("raise", 250), ("fold", 0)]))
    assert (status.waiting_for, status.result, status.hero_net) == (OVER, "you folded", -100)
    status = follow(hand(seat=1, actions=[("fold", 0)]))
    assert (status.result, status.hero_net) == ("the opponent folded", 50)


def test_all_in_and_showdown_leave_the_winner_open():
    status = follow(hand(actions=[("raise", 10_000), ("call", 0)]))
    assert (status.waiting_for, status.result, status.hero_net) == (OVER, "all-in and called", None)
    assert status.history[0]["text"] == "raise to 10000 (all-in)"
    line = [("call", 0), ("check", 0)] + [("check", 0)] * 6
    status = follow(hand(actions=line, board="7h8h2cJd3s"))
    assert (status.result, status.pot, status.hero_net) == ("showdown", 200, None)


def test_unequal_stacks_cap_the_raise():
    status = follow(hand(stacks=(3_000, 10_000)))
    assert status.legal.max_raise_to == 3_000
    status = follow(hand(seat=1, stacks=(10_000, 3_000)))
    assert status.legal.max_raise_to == 3_000


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        (dict(actions=[("check", 0)]), "illegal_action"),
        (dict(actions=[("raise", 150)]), "illegal_action"),
        (dict(actions=[("raise", 250), ("call", 0), ("check", 0)]), "missing_board"),
        (dict(actions=[("fold", 0), ("check", 0)]), "hand_over"),
        (dict(actions=[("shove", 0)]), "bad_action"),
        (dict(board="AsKd2c"), "duplicate_card"),
        (dict(stacks=(50, 10_000)), "bad_stacks"),
    ],
)
def test_bad_input_is_reported_not_repaired(kwargs, code):
    with pytest.raises(SpotError) as err:
        follow(hand(**kwargs))
    assert err.value.code == code


def test_the_opponents_placeholder_cards_never_reach_the_reply():
    line = [("raise", 250), ("call", 0), ("check", 0)]
    reply = advise_live(hand(actions=line, board="7h8h2c")).to_dict()
    text = json.dumps(reply)
    assert "hole" not in text
    assert reply["hero"]["stack"] == reply["opponent"]["stack"] == 9_750


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(time_budget=1.0))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def post(url, payload):
    request = urllib.request.Request(url, json.dumps(payload).encode(), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_server_serves_the_page_and_advice(server):
    with urllib.request.urlopen(server + "/", timeout=10) as response:
        assert b"pokerbot" in response.read()
    base = {"hole": "QsQd", "position": "bb", "my_stack": 10_000, "opp_stack": 10_000}
    code, reply = post(server + "/api/hand", base)
    assert (code, reply["waiting_for"]) == (200, "opponent")
    code, reply = post(
        server + "/api/hand", {**base, "actions": [{"kind": "raise", "amount": 250}]}
    )
    assert (code, reply["waiting_for"]) == (200, "hero")
    assert reply["recommendation"]["kind"] in ("call", "raise")
    code, reply = post(server + "/api/hand", {**base, "actions": [{"kind": "check"}]})
    assert (code, reply["error"]) == (400, "illegal_action")
    code, reply = post(server + "/api/hand", {"hole": "QsQd"})
    assert (code, reply["error"]) == (400, "bad_input")


def test_server_multiway(server):
    payload = {"hole": "AsAd", "board": "7h8h2c", "players": 4, "pot": 1000, "to_call": 400}
    code, reply = post(server + "/api/multiway", payload)
    assert code == 200
    assert 0.5 < reply["equity"] < 0.8
    assert reply["text"].startswith(("raise", "call"))
    code, reply = post(server + "/api/multiway", {**payload, "board": "7h8h"})
    assert (code, reply["error"]) == (400, "bad_board")
