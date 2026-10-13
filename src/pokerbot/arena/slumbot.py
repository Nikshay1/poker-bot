"""Play against Slumbot over its public API.

Slumbot is the 2018 Annual Computer Poker Competition champion and is
playable at slumbot.com. Its game is heads-up No-Limit Hold'em with blinds
50/100 and 20,000 chips (200 big blinds) each, reset every hand.

API, as observed on 2026-10-13:

* ``POST /api/new_hand`` with ``{}`` (or ``{"token": ...}`` to continue a
  session) starts a hand.
* ``POST /api/act`` with ``{"token": ..., "incr": "<action>"}`` sends one action.
* Both return ``action`` (the whole hand so far), ``client_pos``,
  ``hole_cards``, ``board`` and ``token``; a finished hand also has
  ``winnings`` in chips and ``bot_hole_cards``.

The action string separates streets with ``/`` and uses ``k`` check, ``c``
call, ``f`` fold and ``bN`` for a bet or raise to N chips on that street,
which is this project's raise-to convention. ``client_pos`` 0 is the big
blind and 1 is the button, the reverse of this project's seat numbers.

These hands are not duplicate pairs, so the interval is over single hands
and is wide. A few hundred hands show that the bot plays complete, legal
hands against a strong opponent; they do not measure a small edge.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request

import numpy as np

from pokerbot.bots.basic import Bot, make_bot
from pokerbot.cards import card_from_str
from pokerbot.game.state import CALL, CHECK, FOLD, RAISE, Action, GameConfig, HandState, Spot

HOST = "https://slumbot.com"
CONFIG = GameConfig(small_blind=50, big_blind=100, stacks=(20_000, 20_000))
_TOKEN = re.compile(r"b\d+|[kcf]")


class SlumbotError(RuntimeError):
    pass


def _post(path: str, payload: dict, timeout: float = 30.0) -> dict:
    request = urllib.request.Request(
        HOST + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    if "error_msg" in data:
        raise SlumbotError(data["error_msg"])
    return data


def parse_actions(text: str) -> list[tuple[str, int]]:
    """Slumbot's action string as a flat list of (kind, amount) in order."""
    out: list[tuple[str, int]] = []
    for street in text.split("/"):
        tokens = _TOKEN.findall(street)
        if "".join(tokens) != street:
            raise SlumbotError(f"cannot parse action string {text!r}")
        for token in tokens:
            if token == "k":
                out.append((CHECK, 0))
            elif token == "c":
                out.append((CALL, 0))
            elif token == "f":
                out.append((FOLD, 0))
            else:
                out.append((RAISE, int(token[1:])))
    return out


def to_incr(action: Action) -> str:
    if action.kind == RAISE:
        return f"b{action.amount}"
    return {FOLD: "f", CHECK: "k", CALL: "c"}[action.kind]


def spot_from_response(data: dict) -> Spot | None:
    """Rebuild the hero's situation from a response, by replaying the whole
    hand through the rules engine. Returns None when the hero has nothing to
    decide (the hand is over, or all the chips are in)."""
    seat = 1 - int(data["client_pos"])
    hole = tuple(card_from_str(c) for c in data["hole_cards"])
    board = tuple(card_from_str(c) for c in data["board"])
    known = set(hole) | set(board)
    filler = [c for c in range(52) if c not in known]
    other = tuple(filler[:2])
    holes = (hole, other) if seat == 0 else (other, hole)
    deck = holes[0] + holes[1] + board + tuple(filler[2 : 2 + 5 - len(board)])
    state = HandState(CONFIG, deck)
    for kind, amount in parse_actions(data["action"]):
        if state.finished:
            raise SlumbotError(f"actions continue after the hand ended: {data['action']!r}")
        state.apply(Action(kind, amount))
    if state.finished or state.to_act != seat:
        return None
    if len(state.board) != len(board):
        raise SlumbotError("the board does not match the betting")
    return state.view(seat)


def play_hand(bot: Bot, token: str | None = None) -> dict:
    """Play one hand. Returns winnings in chips and the session token."""
    data = _post("/api/new_hand", {"token": token} if token else {})
    token = data.get("token", token)
    bot.new_hand()
    decisions = 0
    while "winnings" not in data:
        spot = spot_from_response(data)
        if spot is None:
            raise SlumbotError(f"no decision to make but the hand is not over: {data}")
        action = bot.act(spot)
        if not spot.legal().allows(action):
            raise SlumbotError(f"the bot chose an illegal action {action}")
        decisions += 1
        data = _post("/api/act", {"token": token, "incr": to_incr(action)})
        token = data.get("token", token)
    return {
        "winnings": int(data["winnings"]),
        "baseline_winnings": data.get("baseline_winnings"),
        "action": data["action"],
        "client_pos": data["client_pos"],
        "hole_cards": data["hole_cards"],
        "bot_hole_cards": data.get("bot_hole_cards"),
        "board": data["board"],
        "decisions": decisions,
        "token": token,
    }


def summarise(winnings_chips, **extra) -> dict:
    """Win rate in big blinds per 100 hands with a 95% interval over hands."""
    bb = np.array(winnings_chips, dtype=np.float64) / CONFIG.big_blind
    n = len(bb)
    stdev = float(bb.std(ddof=1)) if n > 1 else float("nan")
    return {
        "hands": n,
        "bb_per_100": float(bb.mean() * 100) if n else float("nan"),
        "ci95": float(1.96 * stdev / math.sqrt(n) * 100) if n > 1 else float("nan"),
        "stdev_hand_bb": stdev,
        **extra,
    }


def summarise_log(log_path: str) -> dict:
    """Summary of every hand in a session log, across runs."""
    with open(log_path) as log:
        records = [json.loads(line) for line in log if line.strip()]
    return summarise([r["winnings"] for r in records])


def run_session(
    bot_name: str,
    hands: int,
    seed: int = 0,
    log_path: str | None = None,
    max_dropped: int = 20,
) -> dict:
    """Play ``hands`` hands and summarise the result.

    A network failure in the middle of a hand cannot be retried safely,
    because the server may or may not have applied the last action. That hand
    is dropped and counted, and play continues in a fresh session. Errors in
    this program's own logic are not caught.
    """
    bot = make_bot(bot_name, seed)
    token = None
    results = []
    dropped = 0
    started = time.perf_counter()
    while len(results) < hands:
        try:
            record = play_hand(bot, token)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            dropped += 1
            token = None
            if dropped > max_dropped:
                raise SlumbotError(f"too many network failures, last: {err}") from err
            time.sleep(2.0)
            continue
        token = record.pop("token")
        results.append(record["winnings"])
        if log_path:
            # Appended hand by hand, so an interrupted session keeps what it played.
            with open(log_path, "a") as log:
                log.write(json.dumps(record) + "\n")
    return summarise(
        results, bot=bot_name, dropped_hands=dropped, seconds=time.perf_counter() - started
    )
