"""Heuristic strategy before the flop.

Unopened pots use a threshold over hands ranked by equity against a random
hand. Facing a raise, each hand is judged by its equity against the raiser's
current range: re-raise the strongest, call when pot odds are met after an
allowance for how much equity a call realizes, fold the rest. When calling
ends the betting (an all-in), the pot-odds rule is exact and no allowance
is applied.
"""

from __future__ import annotations

import numpy as np

from pokerbot.cards import COMBO_CLASS
from pokerbot.game.state import RAISE, Spot
from pokerbot.policy.base import (
    B_CALL,
    B_FOLD,
    B_RAISE,
    PolicyConfig,
    Strategy,
    sized,
    take_by_order,
)
from pokerbot.preflop_table import class_equity_vs_range, class_weights, equity_vs_random


def preflop_strategy(
    spot: Spot, own_range: np.ndarray, opp_range: np.ndarray, cfg: PolicyConfig
) -> Strategy:
    legal = spot.legal()
    me, opp = spot.seat, 1 - spot.seat
    own = class_weights(own_range)
    total = own.sum()
    pot, to_call = spot.pot, legal.call_amount
    n_raises = sum(1 for street, _, kind, _ in spot.history if street == 0 and kind == RAISE)

    probs = np.zeros((169, 3))
    raise_to = None
    equity = None
    notes: list[str] = []

    if n_raises == 0 and legal.can_raise:
        order = np.argsort(-equity_vs_random(), kind="stable")
        if to_call > 0:
            # Button, first to act: open or fold.
            raise_to = sized(spot, legal, cfg.open_size_bb * spot.big_blind, cfg)
            take = take_by_order(own, order, cfg.open_frac * total)
            probs[:, B_RAISE] = take
            probs[:, B_FOLD] = 1.0 - take
            notes.append(f"opens the top {cfg.open_frac:.0%} of hands")
        else:
            # Big blind after a limp: raise the best hands, check the rest.
            raise_to = sized(spot, legal, cfg.limp_raise_size_bb * spot.big_blind, cfg)
            take = take_by_order(own, order, cfg.limp_raise_frac * total)
            probs[:, B_RAISE] = take
            probs[:, B_CALL] = 1.0 - take
            notes.append(f"raises the top {cfg.limp_raise_frac:.0%} of hands after a limp")
    elif to_call == 0:
        probs[:, B_CALL] = 1.0
    else:
        equity = class_equity_vs_range(opp_range)
        need = to_call / (pot + to_call)
        if not legal.can_raise:
            # Calling ends the betting, so equity against pot odds is exact.
            call = equity >= need
            probs[:, B_CALL] = call
            probs[:, B_FOLD] = ~call
            notes.append(f"calling ends the betting: needs equity of {need:.1%}")
        else:
            in_position = me == 0
            realization = cfg.realization_ip if in_position else cfg.realization_oop
            factor = cfg.reraise_factor_ip if in_position else cfg.reraise_factor_oop
            raise_to = sized(spot, legal, factor * spot.street_contrib[opp], cfg)
            value = (own > 0) & (equity >= cfg.reraise_value_equity)
            call = (equity * realization >= need) & ~value
            candidates = (own > 0) & ~value & ~call
            bluff = take_by_order(
                own * candidates,
                np.argsort(-equity, kind="stable"),
                own[value].sum() * cfg.reraise_bluff_ratio,
            )
            probs[:, B_RAISE] = value + bluff
            probs[:, B_CALL] = call
            probs[:, B_FOLD] = 1.0 - probs[:, B_RAISE] - probs[:, B_CALL]
            notes.append(
                f"needs equity of {need:.1%} to call, with {realization:.0%} of it realized"
            )

    return Strategy(
        probs=np.clip(probs, 0.0, 1.0)[COMBO_CLASS],
        raise_to=raise_to,
        equity=None if equity is None else equity[COMBO_CLASS],
        notes=notes,
    )
