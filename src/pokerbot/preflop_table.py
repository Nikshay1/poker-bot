"""The precomputed 169 x 169 preflop all-in equity table.

Built by ``scripts/build_preflop_equity.py`` (Monte Carlo, 200,000 deals per
pair of classes, standard error about 0.001 per entry).
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np

from pokerbot.cards import CLASS_SIZE, COMBO_CLASS, N_CLASSES


@lru_cache(maxsize=1)
def preflop_equity() -> np.ndarray:
    """``table[i, j]``: equity of class i all-in before the flop against class j."""
    path = resources.files("pokerbot") / "data" / "preflop_equity.npy"
    with resources.as_file(path) as file:
        table = np.load(file).astype(np.float64)
    table.setflags(write=False)
    return table


def class_weights(range_weights: np.ndarray) -> np.ndarray:
    """Collapse a 1326-combo range to total weight per preflop class."""
    return np.bincount(COMBO_CLASS, weights=range_weights, minlength=N_CLASSES)


@lru_cache(maxsize=1)
def equity_vs_random() -> np.ndarray:
    """Each class's equity against a uniformly random hand."""
    table = preflop_equity()
    return table @ CLASS_SIZE / CLASS_SIZE.sum()


def class_equity_vs_range(opp_range: np.ndarray) -> np.ndarray:
    """Each class's all-in equity against a range.

    This uses class-level weights, so it ignores the small effect of the
    hero's own cards removing combos from the opponent's range.
    """
    weights = class_weights(opp_range)
    total = weights.sum()
    if total <= 0:
        raise ValueError("empty range")
    return preflop_equity() @ weights / total
