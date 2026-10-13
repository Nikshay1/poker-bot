"""Heads-up No-Limit Hold'em rules.

Seat 0 is the button and small blind: first to act before the flop, last
after it. Seat 1 is the big blind. Every bet or raise is a "raise to" amount:
the seat's total contribution on the current street after the action.

``legal_actions`` is the single source of truth for what is allowed. The
engine (``HandState``) and the view handed to bots (``Spot``) both use it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pokerbot.evaluator import evaluate

FOLD, CHECK, CALL, RAISE = "fold", "check", "call", "raise"
STREET_NAMES = ["preflop", "flop", "turn", "river"]
_BOARD_CARDS = [0, 3, 4, 5]


class SpotError(ValueError):
    """Invalid or contradictory input. ``code`` is a stable machine-readable tag."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def as_dict(self) -> dict:
        return {"error": self.code, "message": str(self)}


@dataclass(frozen=True)
class Action:
    kind: str
    amount: int = 0  # raise: the raise-to amount; call: chips added; otherwise 0

    def __str__(self) -> str:
        if self.kind == RAISE:
            return f"raise to {self.amount}"
        if self.kind == CALL:
            return f"call {self.amount}"
        return self.kind


@dataclass(frozen=True)
class Legal:
    can_fold: bool
    can_check: bool
    call_amount: int  # 0 when there is nothing to call
    min_raise_to: int | None  # None when raising is not allowed
    max_raise_to: int | None  # the all-in amount

    @property
    def can_raise(self) -> bool:
        return self.min_raise_to is not None

    def allows(self, action: Action) -> bool:
        if action.kind == FOLD:
            return self.can_fold
        if action.kind == CHECK:
            return self.can_check
        if action.kind == CALL:
            return self.call_amount > 0 and action.amount in (0, self.call_amount)
        if action.kind == RAISE:
            return self.can_raise and self.min_raise_to <= action.amount <= self.max_raise_to
        return False

    def clamp_raise(self, raise_to: int) -> int:
        return int(min(max(raise_to, self.min_raise_to), self.max_raise_to))

    def passive(self) -> Action:
        """Check if possible, otherwise fold. Always legal."""
        return Action(CHECK) if self.can_check else Action(FOLD)


def legal_actions(street_contrib, stacks, seat: int, last_raise: int, big_blind: int) -> Legal:
    opp = 1 - seat
    to_call = street_contrib[opp] - street_contrib[seat]
    call_amount = min(to_call, stacks[seat])
    min_to = max_to = None
    # A raise needs chips beyond the call and an opponent who can still act.
    if stacks[seat] > to_call and stacks[opp] > 0:
        # Chips the opponent cannot match are never put in.
        max_to = min(street_contrib[seat] + stacks[seat], street_contrib[opp] + stacks[opp])
        min_to = min(street_contrib[opp] + max(last_raise, big_blind), max_to)
    return Legal(
        can_fold=to_call > 0,
        can_check=to_call == 0,
        call_amount=call_amount,
        min_raise_to=min_to,
        max_raise_to=max_to,
    )


@dataclass(frozen=True)
class Spot:
    """Everything one player may know when it is their turn to act."""

    hole: tuple[int, int]
    board: tuple[int, ...]
    seat: int
    stacks: tuple[int, int]  # chips behind, by seat
    street_contrib: tuple[int, int]
    total_contrib: tuple[int, int]
    small_blind: int = 50
    big_blind: int = 100
    last_raise: int = 100
    history: tuple[tuple[int, int, str, int], ...] = ()  # (street, seat, kind, amount)
    assumptions: tuple[str, ...] = ()
    # Positions in ``history`` that were filled in to complete partial input.
    # They are not observations and must not be used as evidence about ranges.
    unobserved: tuple[int, ...] = ()

    @property
    def street(self) -> int:
        return _BOARD_CARDS.index(len(self.board))

    @property
    def pot(self) -> int:
        return self.total_contrib[0] + self.total_contrib[1]

    @property
    def to_call(self) -> int:
        return self.legal().call_amount

    @property
    def effective_stack(self) -> int:
        """Chips the hero can still win or lose beyond what is already in."""
        return min(self.stacks)

    def legal(self) -> Legal:
        return legal_actions(
            self.street_contrib, self.stacks, self.seat, self.last_raise, self.big_blind
        )

    def validate(self) -> None:
        if len(self.board) not in _BOARD_CARDS:
            raise SpotError("bad_board", "the board must have 0, 3, 4 or 5 cards")
        if len(self.hole) != 2:
            raise SpotError("bad_hole", "the hero needs exactly two hole cards")
        cards = list(self.hole) + list(self.board)
        if any(not 0 <= c < 52 for c in cards):
            raise SpotError("bad_card", "card out of range")
        if len(set(cards)) != len(cards):
            raise SpotError("duplicate_card", "the same card appears twice")
        if self.seat not in (0, 1):
            raise SpotError("bad_seat", "heads-up seats are 0 (button) and 1 (big blind)")
        if self.small_blind <= 0 or self.big_blind < self.small_blind:
            raise SpotError("bad_blinds", "blinds must be positive with small <= big")
        if min(self.stacks) < 0 or min(self.street_contrib) < 0:
            raise SpotError("negative_chips", "stacks and contributions cannot be negative")
        if any(t < s for t, s in zip(self.total_contrib, self.street_contrib)):
            raise SpotError("bad_contrib", "total contribution is below the street contribution")
        opp = 1 - self.seat
        if self.street_contrib[self.seat] > self.street_contrib[opp]:
            raise SpotError("not_to_act", "the hero has put in more than the opponent this street")
        if self.stacks[self.seat] == 0:
            raise SpotError("no_decision", "the hero is all-in and has nothing to decide")


@dataclass(frozen=True)
class GameConfig:
    small_blind: int = 50
    big_blind: int = 100
    stacks: tuple[int, int] = (10_000, 10_000)  # 100 big blinds each

    def __post_init__(self):
        if self.small_blind <= 0 or self.big_blind < self.small_blind:
            raise ValueError("blinds must be positive with small <= big")
        if min(self.stacks) < self.big_blind:
            raise ValueError("each stack must cover the big blind")


@dataclass
class HandState:
    """One hand from the blinds to the payoff.

    ``deck`` fixes the deal: cards 0-1 go to seat 0, cards 2-3 to seat 1 and
    cards 4-8 are the board in order. Passing the same deck twice with the
    players swapped gives a duplicate pair.
    """

    config: GameConfig
    deck: tuple[int, ...]
    stacks: list[int] = field(init=False)
    street_contrib: list[int] = field(init=False)
    total_contrib: list[int] = field(init=False)
    street: int = field(init=False, default=0)
    board: list[int] = field(init=False, default_factory=list)
    to_act: int = field(init=False, default=0)
    last_raise: int = field(init=False)
    acted: list[bool] = field(init=False)
    history: list[tuple[int, int, str, int]] = field(init=False, default_factory=list)
    finished: bool = field(init=False, default=False)
    payoffs: tuple[int, int] | None = field(init=False, default=None)
    folded: int | None = field(init=False, default=None)

    def __post_init__(self):
        deck = tuple(int(c) for c in self.deck[:9])
        if len(deck) != 9 or len(set(deck)) != 9:
            raise ValueError("a deal needs nine different cards")
        self.deck = deck
        cfg = self.config
        self._hole = ((deck[0], deck[1]), (deck[2], deck[3]))
        self.street_contrib = [cfg.small_blind, cfg.big_blind]
        self.total_contrib = [cfg.small_blind, cfg.big_blind]
        self.stacks = [cfg.stacks[0] - cfg.small_blind, cfg.stacks[1] - cfg.big_blind]
        self.last_raise = cfg.big_blind
        self.acted = [False, False]

    def hole(self, seat: int) -> tuple[int, int]:
        return self._hole[seat]

    @property
    def pot(self) -> int:
        return self.total_contrib[0] + self.total_contrib[1]

    def legal(self) -> Legal:
        return legal_actions(
            self.street_contrib, self.stacks, self.to_act, self.last_raise, self.config.big_blind
        )

    def view(self, seat: int | None = None) -> Spot:
        """What ``seat`` may know. The opponent's hole cards and the undealt
        board are not part of it."""
        seat = self.to_act if seat is None else seat
        return Spot(
            hole=self._hole[seat],
            board=tuple(self.board),
            seat=seat,
            stacks=tuple(self.stacks),
            street_contrib=tuple(self.street_contrib),
            total_contrib=tuple(self.total_contrib),
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            last_raise=self.last_raise,
            history=tuple(self.history),
        )

    def apply(self, action: Action) -> None:
        if self.finished:
            raise ValueError("the hand is over")
        legal = self.legal()
        if not legal.allows(action):
            raise ValueError(f"illegal action {action} for seat {self.to_act}; allowed: {legal}")
        seat, opp = self.to_act, 1 - self.to_act
        self.acted[seat] = True

        if action.kind == FOLD:
            self.history.append((self.street, seat, FOLD, 0))
            self.folded = seat
            self._finish_fold(seat)
            return
        if action.kind == CHECK:
            self.history.append((self.street, seat, CHECK, 0))
        elif action.kind == CALL:
            self._put_in(seat, legal.call_amount)
            self.history.append((self.street, seat, CALL, legal.call_amount))
            # A call for less than the bet returns the unmatched part.
            excess = self.street_contrib[opp] - self.street_contrib[seat]
            if excess > 0:
                self._put_in(opp, -excess)
        else:
            increment = action.amount - self.street_contrib[opp]
            # Only a full raise sets a new minimum; a short all-in does not.
            self.last_raise = max(self.last_raise, increment)
            self._put_in(seat, action.amount - self.street_contrib[seat])
            self.history.append((self.street, seat, RAISE, action.amount))

        # The round ends once the bets are matched and both have acted, or
        # matched with someone all-in, which leaves nothing to decide.
        matched = self.street_contrib[0] == self.street_contrib[1]
        if matched and (all(self.acted) or min(self.stacks) == 0):
            self._close_round()
        else:
            self.to_act = opp

    def _put_in(self, seat: int, chips: int) -> None:
        self.stacks[seat] -= chips
        self.street_contrib[seat] += chips
        self.total_contrib[seat] += chips

    def _close_round(self) -> None:
        all_in = min(self.stacks) == 0
        if self.street == 3 or all_in:
            self.board = list(self.deck[4:9])
            self._finish_showdown()
            return
        self.street += 1
        self.board = list(self.deck[4 : 4 + _BOARD_CARDS[self.street]])
        self.street_contrib = [0, 0]
        self.acted = [False, False]
        self.last_raise = self.config.big_blind
        self.to_act = 1

    def _finish_fold(self, folder: int) -> None:
        lost = self.total_contrib[folder]
        self.payoffs = (-lost, lost) if folder == 0 else (lost, -lost)
        self.finished = True

    def _finish_showdown(self) -> None:
        board = np.array(self.board, dtype=np.int64)
        s0 = evaluate(np.concatenate([board, np.array(self._hole[0], dtype=np.int64)]))
        s1 = evaluate(np.concatenate([board, np.array(self._hole[1], dtype=np.int64)]))
        if s0 > s1:
            self.payoffs = (self.total_contrib[1], -self.total_contrib[1])
        elif s1 > s0:
            self.payoffs = (-self.total_contrib[0], self.total_contrib[0])
        else:
            self.payoffs = (0, 0)
        self.finished = True
