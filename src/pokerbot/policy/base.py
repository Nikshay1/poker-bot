"""Shared pieces of the heuristic policy.

A policy answers for a whole range at once: ``Strategy.probs[k, b]`` is the
probability that combo ``k`` takes action bucket ``b``. The same object gives
the hero's decision (one row) and the likelihood used to update a range
after an observed action (one column).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pokerbot.game.state import CALL, CHECK, FOLD, RAISE, Action, Legal, Spot

# Action buckets.
B_FOLD, B_CALL, B_RAISE = 0, 1, 2  # B_CALL covers check and call
BUCKET_NAMES = ["fold", "check/call", "bet/raise"]


def bucket_of(kind: str) -> int:
    if kind == FOLD:
        return B_FOLD
    if kind in (CHECK, CALL):
        return B_CALL
    if kind == RAISE:
        return B_RAISE
    raise ValueError(f"unknown action kind {kind!r}")


@dataclass(frozen=True)
class PolicyConfig:
    """Every tunable number of the heuristic advisor, in one place.

    ``version`` names the fixed model used for range updates; change it when
    any value changes so results stay traceable.
    """

    version: str = "h1"

    # Before the flop.
    open_frac: float = 0.82  # share of hands the button opens
    open_size_bb: float = 2.5
    limp_raise_frac: float = 0.35  # share of hands the big blind raises after a limp
    limp_raise_size_bb: float = 3.5
    reraise_value_equity: float = 0.58  # equity against the raiser's range to re-raise for value
    reraise_bluff_ratio: float = 0.4  # bluff weight per unit of value weight
    reraise_factor_ip: float = 3.0
    reraise_factor_oop: float = 3.5
    realization_ip: float = 1.0  # share of raw equity a call realizes, in position
    realization_oop: float = 0.8  # and out of position

    # After the flop. Tuples are indexed by street: preflop, flop, turn, river.
    bet_frac: float = 0.66  # bet size as a fraction of the pot
    raise_factor: float = 3.0  # raise to this multiple of the bet faced
    value_raise_equity: float = 0.80
    bluff_ceiling: float = 0.40  # only hands below this equity are used as bluffs
    bluff_mult: tuple[float, ...] = (1.0, 2.0, 1.5, 1.0)
    call_margin: tuple[float, ...] = (0.0, 0.03, 0.02, 0.0)
    mdf_guard: float = 0.65  # defend at least this share of the minimum defence frequency

    # Sizing and range updates.
    commit_frac: float = 0.4  # a bet using this share of the stack becomes all-in
    epsilon: float = 0.10  # weight of the uniform component in the action likelihood


@dataclass
class Strategy:
    probs: np.ndarray  # (1326, 3)
    raise_to: int | None  # concrete size of the bet/raise bucket, if it is available
    equity: np.ndarray | None = None  # per-combo equity against the opponent's range
    notes: list[str] = field(default_factory=list)

    def action_for(self, bucket: int, legal: Legal) -> Action:
        """Turn a bucket into a legal action."""
        if bucket == B_RAISE and legal.can_raise and self.raise_to is not None:
            return Action(RAISE, legal.clamp_raise(self.raise_to))
        if bucket == B_FOLD and legal.can_fold:
            return Action(FOLD)
        if legal.can_check:
            return Action(CHECK)
        return Action(CALL, legal.call_amount)


def take_by_order(weights: np.ndarray, order: np.ndarray, budget: float) -> np.ndarray:
    """Walk the entries in ``order`` and take each one fully until ``budget``
    of weight is used up; the entry on the boundary is taken in part.

    Returns the fraction taken of every entry.
    """
    out = np.zeros_like(weights, dtype=np.float64)
    if budget <= 0:
        return out
    w = weights[order]
    before = np.cumsum(w) - w
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(w > 0, np.clip((budget - before) / w, 0.0, 1.0), 0.0)
    out[order] = frac
    return out


def sized(spot: Spot, legal: Legal, target_to: float, cfg: PolicyConfig) -> int:
    """Clamp a bet to the legal range; go all-in once it would commit the stack."""
    raise_to = legal.clamp_raise(int(round(target_to)))
    added = raise_to - spot.street_contrib[spot.seat]
    can_add = legal.max_raise_to - spot.street_contrib[spot.seat]
    if added >= cfg.commit_frac * can_add:
        raise_to = legal.max_raise_to
    return int(raise_to)


def last_bet_risk(spot: Spot) -> int:
    """Chips the opponent added with the bet or raise the hero now faces."""
    opp = 1 - spot.seat
    before = 0
    if spot.street == 0:
        before = spot.small_blind if opp == 0 else spot.big_blind
    current = before
    for street, seat, kind, amount in spot.history:
        if street != spot.street or seat != opp:
            continue
        if kind == RAISE:
            before, current = current, amount
        elif kind == CALL:
            before = current = current + amount
    return max(current - before, 0)
