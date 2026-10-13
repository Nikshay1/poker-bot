"""Follow one real hand as it is played and advise the hero at each decision.

The caller sends everything seen so far: the hero's cards, the board cards
dealt, and every action in order. ``follow`` runs that through the rules
engine and says what is needed next: a decision from the hero, an action from
the opponent, more board cards, or nothing because the hand is over. When the
hero is to act, ``advise_live`` adds the recommendation.

Nothing is assumed here. Every action is one the user saw, so the whole
history narrows the ranges, unlike the filler in ``spot_from_partial``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pokerbot.advisor import Recommendation, SolverConfig, advise
from pokerbot.cards import parse_cards
from pokerbot.game.state import (
    CALL,
    CHECK,
    FOLD,
    RAISE,
    STREET_NAMES,
    Action,
    GameConfig,
    HandState,
    Legal,
    Spot,
    SpotError,
)

HERO, OPPONENT, BOARD, OVER = "hero", "opponent", "board", "over"
_BOARD_CARDS = [0, 3, 4, 5]
_KINDS = (FOLD, CHECK, CALL, RAISE)


@dataclass(frozen=True)
class LiveHand:
    """What the user has entered so far. Seat 0 is the button."""

    hole: tuple[int, int]
    seat: int
    stacks: tuple[int, int]  # by seat, at the start of the hand, before the blinds
    small_blind: int = 50
    big_blind: int = 100
    board: tuple[int, ...] = ()
    actions: tuple[tuple[str, int], ...] = ()  # (kind, raise-to amount) in the order played


@dataclass
class LiveStatus:
    seat: int  # the hero's seat
    waiting_for: str  # HERO, OPPONENT, BOARD or OVER
    street: int
    pot: int
    stacks: tuple[int, int]
    street_contrib: tuple[int, int]
    total_contrib: tuple[int, int]
    history: list[dict] = field(default_factory=list)
    legal: Legal | None = None  # for whoever is to act
    cards_needed: int = 0  # board cards still to enter, when waiting for the board
    result: str = ""  # how the hand ended
    hero_net: int | None = None  # known only when the hand ended in a fold
    spot: Spot | None = None  # the hero's view, when the hero is to act
    recommendation: Recommendation | None = None

    def to_dict(self) -> dict:
        me, opp = self.seat, 1 - self.seat

        def side(seat):
            return {
                "stack": self.stacks[seat],
                "street_contrib": self.street_contrib[seat],
                "total_contrib": self.total_contrib[seat],
            }

        legal = self.legal
        return {
            "waiting_for": self.waiting_for,
            "street": STREET_NAMES[self.street],
            "pot": self.pot,
            "hero": side(me),
            "opponent": side(opp),
            "legal": legal
            and {
                "can_fold": legal.can_fold,
                "can_check": legal.can_check,
                "call_amount": legal.call_amount,
                "min_raise_to": legal.min_raise_to,
                "max_raise_to": legal.max_raise_to,
            },
            "cards_needed": self.cards_needed,
            "result": self.result,
            "hero_net": self.hero_net,
            "history": self.history,
            "recommendation": self.recommendation and _advice(self.recommendation, self.spot),
        }


def _advice(rec: Recommendation, spot: Spot) -> dict:
    """The recommendation with every option as a concrete, enterable action."""
    legal = spot.legal()
    options = []
    for name, p in rec.probabilities.items():
        # Names are "fold", "check", "call 400", "raise to 900", or a bare
        # "call" / "raise" from the heuristic layer.
        kind, _, tail = name.partition(" ")
        if kind == RAISE:
            action = Action(RAISE, int(tail.split()[-1]) if tail else rec.raise_to)
        else:
            action = Action(kind, legal.call_amount if kind == CALL else 0)
        options.append(
            {
                "kind": action.kind,
                "amount": action.amount,
                "text": describe(action, spot),
                "probability": p,
                "ev": rec.ev.get(name),
            }
        )
    return {
        "kind": rec.action.kind,
        "amount": rec.action.amount,
        "text": describe(rec.action, spot),
        "options": options,
        "equity": rec.equity,
        "layer": rec.layer,
        "notes": rec.notes,
        "iterations": rec.iterations,
        "seconds": round(rec.seconds, 4),
    }


def hand_from_payload(data: dict) -> LiveHand:
    """Build a ``LiveHand`` from the JSON the web page sends."""
    try:
        position = data.get("position", "button")
        if position not in ("button", "bb"):
            raise SpotError("bad_position", "position must be 'button' or 'bb'")
        seat = 0 if position == "button" else 1
        stacks = [0, 0]
        stacks[seat], stacks[1 - seat] = int(data["my_stack"]), int(data["opp_stack"])
        actions = tuple(
            (str(a["kind"]), int(a.get("amount") or 0)) for a in data.get("actions", [])
        )
        return LiveHand(
            hole=tuple(parse_cards(str(data["hole"]))),
            seat=seat,
            stacks=tuple(stacks),
            small_blind=int(data.get("small_blind", 50)),
            big_blind=int(data.get("big_blind", 100)),
            board=tuple(parse_cards(str(data.get("board", "")))),
            actions=actions,
        )
    except SpotError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as err:
        raise SpotError("bad_input", f"could not read the hand: {err}") from err


def describe(action: Action, before: Spot) -> str:
    """An action in table language: a first bet after the flop is a "bet"."""
    legal = before.legal()
    if action.kind == CALL:
        return f"call {legal.call_amount}"
    if action.kind != RAISE:
        return action.kind
    opening = before.street > 0 and legal.call_amount == 0
    text = f"bet {action.amount}" if opening else f"raise to {action.amount}"
    return f"{text} (all-in)" if action.amount == legal.max_raise_to else text


def follow(hand: LiveHand) -> LiveStatus:
    """Replay what was entered and report what the hand is waiting for."""
    if hand.seat not in (0, 1):
        raise SpotError("bad_seat", "heads-up seats are 0 (button) and 1 (big blind)")
    if len(hand.hole) != 2:
        raise SpotError("bad_hole", "the hero needs exactly two hole cards")
    if len(hand.board) > 5:
        raise SpotError("bad_board", "the board has at most five cards")
    known = list(hand.hole) + list(hand.board)
    if any(not 0 <= c < 52 for c in known):
        raise SpotError("bad_card", "card out of range")
    if len(set(known)) != len(known):
        raise SpotError("duplicate_card", "the same card appears twice")
    try:
        config = GameConfig(hand.small_blind, hand.big_blind, hand.stacks)
    except ValueError as err:
        raise SpotError("bad_stacks", str(err)) from err

    # The opponent's cards and the undealt board are placeholders. Only public
    # fields and the hero's own view are ever read from this state.
    filler = [c for c in range(52) if c not in known]
    holes = [tuple(filler[:2]), tuple(filler[:2])]
    holes[hand.seat] = tuple(hand.hole)
    deck = holes[0] + holes[1] + tuple(hand.board) + tuple(filler[2 : 7 - len(hand.board)])
    state = HandState(config, deck)

    history = []
    for kind, amount in hand.actions:
        if kind not in _KINDS:
            raise SpotError("bad_action", f"unknown action {kind!r}")
        if state.finished:
            raise SpotError("hand_over", "an action was entered after the hand ended")
        if len(hand.board) < _BOARD_CARDS[state.street]:
            raise SpotError("missing_board", "an action was entered before the board cards")
        before = state.view()
        legal = before.legal()
        action = Action(kind, legal.call_amount if kind == CALL else int(amount))
        if not legal.allows(action):
            raise SpotError("illegal_action", _why_illegal(action, legal))
        history.append(
            {
                "street": STREET_NAMES[state.street],
                "who": HERO if state.to_act == hand.seat else OPPONENT,
                "kind": kind,
                "amount": action.amount,
                "text": describe(action, before),
            }
        )
        state.apply(action)

    status = LiveStatus(
        seat=hand.seat,
        waiting_for=OVER,
        street=state.street,
        pot=state.pot,
        stacks=tuple(state.stacks),
        street_contrib=tuple(state.street_contrib),
        total_contrib=tuple(state.total_contrib),
        history=history,
    )
    if state.finished:
        if state.folded is None:
            # The showdown was run on placeholder cards, so its result is not read.
            all_in = min(state.stacks) == 0
            status.result = "all-in and called" if all_in else "showdown"
        else:
            status.result = "you folded" if state.folded == hand.seat else "the opponent folded"
            status.hero_net = state.payoffs[hand.seat]
        return status
    missing = _BOARD_CARDS[state.street] - len(hand.board)
    if missing > 0:
        status.waiting_for, status.cards_needed = BOARD, missing
        return status
    status.legal = state.legal()
    if state.to_act == hand.seat:
        status.waiting_for, status.spot = HERO, state.view(hand.seat)
    else:
        status.waiting_for = OPPONENT
    return status


def advise_live(hand: LiveHand, solver: SolverConfig | None = None) -> LiveStatus:
    """``follow``, plus the recommendation when the hero is to act."""
    status = follow(hand)
    if status.spot is not None:
        status.recommendation = advise(status.spot, solver=solver)
    return status


def _why_illegal(action: Action, legal: Legal) -> str:
    if action.kind == RAISE and legal.can_raise:
        return (
            f"a bet or raise here must be between {legal.min_raise_to} and "
            f"{legal.max_raise_to} in total for this street, not {action.amount}"
        )
    if action.kind == RAISE:
        return "no bet or raise is possible here"
    if action.kind == CHECK:
        return f"there is a bet of {legal.call_amount} to call, so a check is not possible"
    if action.kind == CALL:
        return "there is nothing to call"
    return "there is no bet to fold to; check instead"
