"""Public ranges and how they are updated.

A range is a vector of 1326 non-negative weights, one per hole-card combo.
Both players' ranges are public knowledge: they depend only on the board and
the betting, never on anyone's actual cards. After each action the actor's
range is updated by Bayes' rule,

    weight(hand) *= P(action | hand, public state),

where the likelihood comes from one fixed policy model mixed with a small
uniform component, so a surprising action shrinks the weight of unlikely
hands without ever removing them.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pokerbot.cards import N_COMBOS, blocked_combos
from pokerbot.game.state import Action, GameConfig, HandState, Spot, SpotError
from pokerbot.policy.base import PolicyConfig, Strategy, bucket_of
from pokerbot.policy.postflop import postflop_strategy
from pokerbot.policy.preflop import preflop_strategy

StrategyFn = Callable[[Spot, np.ndarray, np.ndarray, PolicyConfig], Strategy]


def heuristic_strategy(
    spot: Spot, own_range: np.ndarray, opp_range: np.ndarray, cfg: PolicyConfig
) -> Strategy:
    if spot.street == 0:
        return preflop_strategy(spot, own_range, opp_range, cfg)
    return postflop_strategy(spot, own_range, opp_range, cfg)


def replay(spot: Spot) -> list[tuple[Spot, str, int]]:
    """Rebuild the public state before each action in ``spot.history``.

    Returns ``(state before the action, kind, amount)`` for every past action,
    by running the history through the rules engine. The rebuilt spots carry
    placeholder hole cards and must only be used for their public fields. A
    history that is illegal, or does not lead to ``spot``, is an error.
    """
    start = tuple(spot.stacks[i] + spot.total_contrib[i] for i in (0, 1))
    try:
        config = GameConfig(spot.small_blind, spot.big_blind, start)
    except ValueError as err:
        raise SpotError("bad_stacks", str(err)) from err
    known = set(spot.hole) | set(spot.board)
    filler = [c for c in range(52) if c not in known]
    missing = 5 - len(spot.board)
    deck = tuple(filler[:4]) + tuple(spot.board) + tuple(filler[4 : 4 + missing])
    state = HandState(config, deck)

    steps = []
    for street, seat, kind, amount in spot.history:
        if state.finished or state.street != street or state.to_act != seat:
            raise SpotError("bad_history", "the betting history is out of order")
        steps.append((state.view(), kind, amount))
        try:
            state.apply(Action(kind, amount))
        except ValueError as err:
            raise SpotError("bad_history", str(err)) from err
    consistent = (
        not state.finished
        and state.to_act == spot.seat
        and len(state.board) == len(spot.board)
        and tuple(state.street_contrib) == spot.street_contrib
        and tuple(state.total_contrib) == spot.total_contrib
        and tuple(state.stacks) == spot.stacks
    )
    if not consistent:
        raise SpotError("bad_history", "the betting history does not lead to this situation")
    return steps


class RangeTracker:
    """Both players' public ranges over one hand."""

    def __init__(
        self,
        cfg: PolicyConfig | None = None,
        strategy_fn: StrategyFn = heuristic_strategy,
        ranges: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.cfg = cfg or PolicyConfig()
        self.strategy_fn = strategy_fn
        if ranges is None:
            self.ranges = [np.ones(N_COMBOS), np.ones(N_COMBOS)]
        else:
            self.ranges = [np.array(r, dtype=np.float64) for r in ranges]
        self._board: tuple[int, ...] = ()
        self._cache: dict[int, Strategy] = {}
        self.n_seen = 0  # how much of the history has been processed

    def _sync_board(self, board: tuple[int, ...]) -> None:
        if board != self._board:
            blocked = blocked_combos(board)
            for r in self.ranges:
                r[blocked] = 0.0
            self._board = board

    def strategy(self, spot: Spot) -> Strategy:
        """The model strategy for the player to act, over that player's whole range."""
        key = len(spot.history)
        if key not in self._cache:
            self._sync_board(spot.board)
            own, opp = self.ranges[spot.seat], self.ranges[1 - spot.seat]
            self._cache[key] = self.strategy_fn(spot, own, opp, self.cfg)
        return self._cache[key]

    def observe(self, spot: Spot, kind: str) -> None:
        """Update the actor's range for an action taken at ``spot``."""
        strategy = self.strategy(spot)
        eps = self.cfg.epsilon
        likelihood = (1.0 - eps) * strategy.probs[:, bucket_of(kind)] + eps / 3.0
        updated = self.ranges[spot.seat] * likelihood
        # Keep the scale stable over many updates; only relative weights matter.
        peak = updated.max()
        if peak > 0:
            updated /= peak
        self.ranges[spot.seat] = updated

    def catch_up(self, spot: Spot) -> None:
        """Process the part of ``spot.history`` not seen yet, then sync the board.

        Actions listed in ``spot.unobserved`` were assumed, not seen, so they
        leave the ranges unchanged.
        """
        steps = replay(spot)
        skip = set(spot.unobserved)
        for i in range(self.n_seen, len(steps)):
            before, kind, _ = steps[i]
            if i not in skip:
                self.observe(before, kind)
        self.n_seen = len(steps)
        self._sync_board(spot.board)
