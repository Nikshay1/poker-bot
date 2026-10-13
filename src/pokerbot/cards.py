"""Cards, hole-card combos and preflop hand classes.

A card is the integer ``4 * rank + suit`` with rank 0..12 for 2..A and suit
0..3 for clubs, diamonds, hearts, spades. A combo is an unordered pair of
cards ``c1 < c2`` with index ``c2 * (c2 - 1) // 2 + c1``.
"""

from __future__ import annotations

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"
N_CARDS = 52
N_COMBOS = 1326
N_CLASSES = 169


def card_from_str(text: str) -> int:
    if len(text) != 2:
        raise ValueError(f"not a card: {text!r}")
    rank = RANKS.find(text[0].upper())
    suit = SUITS.find(text[1].lower())
    if rank < 0 or suit < 0:
        raise ValueError(f"not a card: {text!r}")
    return 4 * rank + suit


def card_to_str(card: int) -> str:
    return RANKS[card >> 2] + SUITS[card & 3]


def parse_cards(text: str) -> list[int]:
    """Parse ``"AsKd"``, ``"As Kd"`` or ``"As,Kd"``. Duplicates are an error."""
    compact = text.replace(",", "").replace(" ", "")
    if len(compact) % 2:
        raise ValueError(f"cannot split into cards: {text!r}")
    cards = [card_from_str(compact[i : i + 2]) for i in range(0, len(compact), 2)]
    if len(set(cards)) != len(cards):
        raise ValueError(f"duplicate card in {text!r}")
    return cards


def cards_to_str(cards) -> str:
    return "".join(card_to_str(int(c)) for c in cards)


def combo_index(c1: int, c2: int) -> int:
    if c1 == c2:
        raise ValueError("a combo needs two different cards")
    if c1 > c2:
        c1, c2 = c2, c1
    return c2 * (c2 - 1) // 2 + c1


def class_index(c1: int, c2: int) -> int:
    """Preflop class on a 13x13 grid: pairs on the diagonal, suited hands
    below it (row = high rank), offsuit hands above it (row = low rank)."""
    r1, r2 = c1 >> 2, c2 >> 2
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo or (c1 & 3) == (c2 & 3):
        return hi * 13 + lo
    return lo * 13 + hi


def class_label(index: int) -> str:
    row, col = divmod(index, 13)
    if row == col:
        return RANKS[row] * 2
    if row > col:
        return RANKS[row] + RANKS[col] + "s"
    return RANKS[col] + RANKS[row] + "o"


def class_from_label(label: str) -> int:
    hi, lo = RANKS.index(label[0].upper()), RANKS.index(label[1].upper())
    if hi < lo:
        hi, lo = lo, hi
    if hi == lo:
        return hi * 13 + hi
    suited = label[2:].lower() == "s"
    return hi * 13 + lo if suited else lo * 13 + hi


# Combo tables, indexed by combo index.
COMBO_CARDS = np.array([(c1, c2) for c2 in range(N_CARDS) for c1 in range(c2)], dtype=np.int64)
COMBO_CLASS = np.array([class_index(c1, c2) for c1, c2 in COMBO_CARDS], dtype=np.int64)
CLASS_SIZE = np.bincount(COMBO_CLASS, minlength=N_CLASSES).astype(np.float64)
CLASS_LABELS = [class_label(i) for i in range(N_CLASSES)]


def blocked_combos(cards) -> np.ndarray:
    """Boolean mask of the combos that contain any of the given cards."""
    dead = np.zeros(N_CARDS, dtype=bool)
    dead[list(cards)] = True
    return dead[COMBO_CARDS[:, 0]] | dead[COMBO_CARDS[:, 1]]


def combo_to_str(index: int) -> str:
    c1, c2 = COMBO_CARDS[index]
    return card_to_str(int(c2)) + card_to_str(int(c1))
