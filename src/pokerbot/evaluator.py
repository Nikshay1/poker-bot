"""Hand strength for 5 to 7 cards.

``evaluate`` returns one integer: larger is stronger, equal is a tie. The top
bits hold the category, the low 20 bits hold up to five ranks that break ties
inside the category.

The method is direct arithmetic on rank counts and per-suit rank masks, with
no lookup tables, so it runs inside Numba loops.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from pokerbot.cards import COMBO_CARDS, N_COMBOS

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = [
    "high card",
    "pair",
    "two pair",
    "three of a kind",
    "straight",
    "flush",
    "full house",
    "four of a kind",
    "straight flush",
]
CATEGORY_SHIFT = 20
_WHEEL = 0x100F  # A, 5, 4, 3, 2


@njit(cache=True)
def _straight_high(mask):
    """Top rank of the best straight in a 13-bit rank mask, or -1."""
    for high in range(12, 3, -1):
        if (mask >> (high - 4)) & 0x1F == 0x1F:
            return high
    if mask & _WHEEL == _WHEEL:
        return 3
    return -1


@njit(cache=True)
def _top_ranks(mask, k):
    """The k highest ranks in a mask, packed 4 bits each, highest first."""
    packed = 0
    taken = 0
    for rank in range(12, -1, -1):
        if (mask >> rank) & 1:
            packed = (packed << 4) | rank
            taken += 1
            if taken == k:
                break
    return packed


@njit(cache=True)
def evaluate(cards):
    """Strength of the best 5-card hand among 5 to 7 cards."""
    counts = 0  # rank counts, 4 bits per rank
    s0 = s1 = s2 = s3 = 0  # rank mask per suit
    n0 = n1 = n2 = n3 = 0  # cards per suit
    for i in range(len(cards)):
        card = cards[i]
        rank = card >> 2
        suit = card & 3
        counts += 1 << (4 * rank)
        bit = 1 << rank
        if suit == 0:
            s0 |= bit
            n0 += 1
        elif suit == 1:
            s1 |= bit
            n1 += 1
        elif suit == 2:
            s2 |= bit
            n2 += 1
        else:
            s3 |= bit
            n3 += 1

    # With at most 7 cards a flush rules out quads and a full house, so a
    # flush suit can only be a flush or a straight flush.
    flush_mask = 0
    if n0 >= 5:
        flush_mask = s0
    elif n1 >= 5:
        flush_mask = s1
    elif n2 >= 5:
        flush_mask = s2
    elif n3 >= 5:
        flush_mask = s3
    if flush_mask:
        high = _straight_high(flush_mask)
        if high >= 0:
            return (STRAIGHT_FLUSH << CATEGORY_SHIFT) | (high << 16)
        return (FLUSH << CATEGORY_SHIFT) | _top_ranks(flush_mask, 5)

    mask = s0 | s1 | s2 | s3
    quad = trip1 = trip2 = pair1 = pair2 = -1
    for rank in range(12, -1, -1):
        count = (counts >> (4 * rank)) & 15
        if count == 4:
            quad = rank
        elif count == 3:
            if trip1 < 0:
                trip1 = rank
            elif trip2 < 0:
                trip2 = rank
        elif count == 2:
            if pair1 < 0:
                pair1 = rank
            elif pair2 < 0:
                pair2 = rank

    if quad >= 0:
        kicker = _top_ranks(mask & ~(1 << quad), 1)
        return (QUADS << CATEGORY_SHIFT) | (quad << 16) | (kicker << 12)
    if trip1 >= 0 and (trip2 >= 0 or pair1 >= 0):
        return (FULL_HOUSE << CATEGORY_SHIFT) | (trip1 << 16) | (max(trip2, pair1) << 12)
    high = _straight_high(mask)
    if high >= 0:
        return (STRAIGHT << CATEGORY_SHIFT) | (high << 16)
    if trip1 >= 0:
        kickers = _top_ranks(mask & ~(1 << trip1), 2)
        return (TRIPS << CATEGORY_SHIFT) | (trip1 << 16) | (kickers << 8)
    if pair2 >= 0:
        # A third pair, if any, can still supply the kicker.
        kicker = _top_ranks(mask & ~((1 << pair1) | (1 << pair2)), 1)
        return (TWO_PAIR << CATEGORY_SHIFT) | (pair1 << 16) | (pair2 << 12) | (kicker << 8)
    if pair1 >= 0:
        kickers = _top_ranks(mask & ~(1 << pair1), 3)
        return (PAIR << CATEGORY_SHIFT) | (pair1 << 16) | kickers
    return (HIGH_CARD << CATEGORY_SHIFT) | _top_ranks(mask, 5)


@njit(cache=True)
def category(strength):
    return strength >> CATEGORY_SHIFT


@njit(cache=True)
def _combo_strengths(board, combo_cards, out):
    n_board = len(board)
    dead = np.zeros(52, dtype=np.bool_)
    hand = np.empty(n_board + 2, dtype=np.int64)
    for i in range(n_board):
        dead[board[i]] = True
        hand[i] = board[i]
    for i in range(combo_cards.shape[0]):
        c1 = combo_cards[i, 0]
        c2 = combo_cards[i, 1]
        if dead[c1] or dead[c2]:
            out[i] = -1
        else:
            hand[n_board] = c1
            hand[n_board + 1] = c2
            out[i] = evaluate(hand)


def combo_strengths(board) -> np.ndarray:
    """Strength of every hole-card combo on a 3 to 5 card board.

    Combos that share a card with the board get -1.
    """
    out = np.empty(N_COMBOS, dtype=np.int64)
    _combo_strengths(np.asarray(board, dtype=np.int64), COMBO_CARDS, out)
    return out
