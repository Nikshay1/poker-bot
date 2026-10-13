"""Heuristic strategy on the flop, turn and river.

Every hand in the range is scored by its exact equity against the opponent's
current range. The rules come from the solution of the simplest betting game
(a polarized bettor against a bluff-catcher), which gives two numbers for a
bet of ``s`` times the pot:

* the caller must continue with ``1 / (1 + s)`` of its range (the minimum
  defence frequency) or bluffs show an automatic profit;
* the bettor may bluff ``s / (1 + s)`` hands for every value hand.

Applied here: bet for value when the hand beats the middle of the range that
would continue, add bluffs in that ratio, call when pot odds are met, and
never fold more than a share of the minimum defence frequency allows.

These are heuristics. Equity on the flop and turn assumes the hand is
checked to showdown, which misprices draws and position.
"""

from __future__ import annotations

import numpy as np

from pokerbot.equity import range_equity
from pokerbot.game.state import Spot
from pokerbot.policy.base import (
    B_CALL,
    B_FOLD,
    B_RAISE,
    PolicyConfig,
    Strategy,
    last_bet_risk,
    sized,
    take_by_order,
)


def _bluff_order(equity: np.ndarray, street: int) -> np.ndarray:
    # On the river a bluff should be the hand with the least showdown value.
    # Earlier, prefer the candidates with the most equity when called.
    return np.argsort(equity if street == 3 else -equity, kind="stable")


def postflop_strategy(
    spot: Spot, own_range: np.ndarray, opp_range: np.ndarray, cfg: PolicyConfig
) -> Strategy:
    legal = spot.legal()
    me, opp = spot.seat, 1 - spot.seat
    street = spot.street
    raw = range_equity(spot.board, opp_range)
    live = ~np.isnan(raw)
    equity = np.where(live, raw, 0.0)
    weights = np.where(live, own_range, 0.0)
    total = weights.sum()
    pot, to_call = spot.pot, legal.call_amount
    mult = cfg.bluff_mult[street]

    probs = np.zeros((len(equity), 3))
    raise_to = None
    notes: list[str] = []

    if to_call == 0:
        bet = np.zeros(len(equity))
        if legal.can_raise and total > 0:
            raise_to = sized(spot, legal, spot.street_contrib[me] + cfg.bet_frac * pot, cfg)
            size = (raise_to - spot.street_contrib[me]) / pot
            defend = 1.0 / (1.0 + size)
            value_cut = 1.0 - defend / 2.0
            value = (weights > 0) & (equity >= value_cut)
            candidates = (weights > 0) & ~value & (equity < cfg.bluff_ceiling)
            bluff = take_by_order(
                weights * candidates,
                _bluff_order(equity, street),
                weights[value].sum() * size / (1.0 + size) * mult,
            )
            bet = value + bluff
            if street < 3:
                bet = np.where(equity >= cfg.slowplay_equity, 0.0, bet)
            notes.append(
                f"bets {size:.0%} of the pot for value with equity of {value_cut:.0%} or more"
            )
        probs[:, B_RAISE] = bet
        probs[:, B_CALL] = 1.0 - bet
    else:
        need = to_call / (pot + to_call)
        call_ok = (weights > 0) & (equity >= need + cfg.call_margin[street])
        raised = np.zeros(len(equity))
        if legal.can_raise and total > 0:
            raise_to = sized(spot, legal, cfg.raise_factor * spot.street_contrib[opp], cfg)
            size = (raise_to - spot.street_contrib[opp]) / (pot + to_call)
            value = (weights > 0) & (equity >= cfg.value_raise_equity)
            candidates = (weights > 0) & ~value & ~call_ok
            bluff = take_by_order(
                weights * candidates,
                _bluff_order(equity, street),
                weights[value].sum() * size / (1.0 + size) * mult,
            )
            raised = value + bluff
        continuing = np.maximum(raised, call_ok.astype(np.float64))

        # Guard against folding so often that any two cards can bet profitably.
        risk = last_bet_risk(spot) or to_call
        defend = max(1.0 - risk / pot, 0.0)
        shortfall = cfg.mdf_guard * defend * total - (weights * continuing).sum()
        if shortfall > 0:
            folding = (weights > 0) & (continuing == 0)
            extra = take_by_order(weights * folding, np.argsort(-equity, kind="stable"), shortfall)
            continuing = continuing + extra
            notes.append("calls wider than pot odds alone to respect minimum defence")

        probs[:, B_RAISE] = raised
        probs[:, B_CALL] = np.clip(continuing - raised, 0.0, 1.0)
        probs[:, B_FOLD] = 1.0 - probs[:, B_RAISE] - probs[:, B_CALL]
        notes.append(f"needs equity of {need:.1%} to call")

    return Strategy(
        probs=np.clip(probs, 0.0, 1.0),
        raise_to=raise_to,
        equity=np.where(live, raw, np.nan),
        notes=notes,
    )
