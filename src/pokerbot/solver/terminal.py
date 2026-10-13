"""Fast terminal values for a No-Limit Hold'em river.

Both players' hands are indexed by the 1326 hole-card combos. A hand only
meets opponent hands that share no card with it, so every sum below is over
compatible combos. Payoffs are net chips from the start of the hand.

The naive way to get a value for every hand is a 1326 x 1326 matrix product.
Here each terminal costs one pass over the combos:

* fold: every compatible opponent hand pays the same, so the value is that
  amount times the compatible reach, found from one total and 52 per-card
  sums;
* showdown: with hands sorted by strength once per solve, the reach a hand
  beats is a running sum, corrected by per-card running sums.

``tests/test_river.py`` checks both against the matrix product.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from pokerbot.cards import COMBO_CARDS
from pokerbot.evaluator import combo_strengths
from pokerbot.solver.public_tree import Terminal


@njit(cache=True)
def compatible_reach(reach, live, combo_cards, out):
    """For every live combo, the opponent reach on combos sharing no card with it."""
    card = np.zeros(52)
    total = 0.0
    n = reach.shape[0]
    for k in range(n):
        w = reach[k]
        if w != 0.0:
            total += w
            card[combo_cards[k, 0]] += w
            card[combo_cards[k, 1]] += w
    for k in range(n):
        if live[k]:
            out[k] = total - card[combo_cards[k, 0]] - card[combo_cards[k, 1]] + reach[k]
        else:
            out[k] = 0.0


@njit(cache=True)
def showdown_values(order, first_live, strengths, reach, combo_cards, each, out):
    """``each`` times (reach beaten minus reach lost to), for every live combo.

    ``order`` sorts the combos by strength; entries before ``first_live`` are
    blocked by the board.
    """
    n = order.shape[0]
    card_all = np.zeros(52)
    card_less = np.zeros(52)
    card_equal = np.zeros(52)
    weight_all = 0.0
    for t in range(first_live, n):
        k = order[t]
        w = reach[k]
        if w != 0.0:
            weight_all += w
            card_all[combo_cards[k, 0]] += w
            card_all[combo_cards[k, 1]] += w
    for t in range(first_live):
        out[order[t]] = 0.0

    weight_less = 0.0
    i = first_live
    while i < n:
        strength = strengths[order[i]]
        j = i
        weight_equal = 0.0
        while j < n and strengths[order[j]] == strength:
            k = order[j]
            w = reach[k]
            if w != 0.0:
                weight_equal += w
                card_equal[combo_cards[k, 0]] += w
                card_equal[combo_cards[k, 1]] += w
            j += 1
        for t in range(i, j):
            k = order[t]
            a = combo_cards[k, 0]
            b = combo_cards[k, 1]
            w = reach[k]
            win = weight_less - card_less[a] - card_less[b]
            tie = weight_equal - card_equal[a] - card_equal[b] + w
            total = weight_all - card_all[a] - card_all[b] + w
            out[k] = each * (2.0 * win + tie - total)
        for t in range(i, j):
            k = order[t]
            w = reach[k]
            a = combo_cards[k, 0]
            b = combo_cards[k, 1]
            if w != 0.0:
                card_less[a] += w
                card_less[b] += w
            card_equal[a] = 0.0
            card_equal[b] = 0.0
        weight_less += weight_equal
        i = j


class RiverBoard:
    """What every terminal of one river solve shares: strengths and their order."""

    def __init__(self, board):
        if len(board) != 5:
            raise ValueError("a river board has five cards")
        self.board = tuple(int(c) for c in board)
        self.strengths = combo_strengths(self.board)
        self.live = self.strengths >= 0
        self.order = np.argsort(self.strengths, kind="stable")
        self.first_live = int((~self.live).sum())

    def compatible(self, reach: np.ndarray) -> np.ndarray:
        out = np.empty(len(reach))
        compatible_reach(reach, self.live, COMBO_CARDS, out)
        return out


class FoldTerminal(Terminal):
    def __init__(self, board: RiverBoard, folder: int, lost: float):
        self.board, self.folder, self.lost = board, folder, float(lost)

    def utility(self, player: int, reach_opp: np.ndarray) -> np.ndarray:
        sign = -1.0 if player == self.folder else 1.0
        return sign * self.lost * self.board.compatible(reach_opp)


class ShowdownTerminal(Terminal):
    def __init__(self, board: RiverBoard, each: float):
        self.board, self.each = board, float(each)

    def utility(self, player: int, reach_opp: np.ndarray) -> np.ndarray:
        out = np.empty(len(reach_opp))
        b = self.board
        showdown_values(b.order, b.first_live, b.strengths, reach_opp, COMBO_CARDS, self.each, out)
        return out
