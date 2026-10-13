import itertools
import random

import numpy as np
import pytest
from numba import njit

from pokerbot.cards import N_CARDS, parse_cards
from pokerbot.evaluator import (
    CATEGORY_SHIFT,
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    QUADS,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPS,
    TWO_PAIR,
    category,
    combo_strengths,
    evaluate,
)

# Known counts of 5-card hands by category.
FIVE_CARD_COUNTS = {
    HIGH_CARD: 1_302_540,
    PAIR: 1_098_240,
    TWO_PAIR: 123_552,
    TRIPS: 54_912,
    STRAIGHT: 10_200,
    FLUSH: 5_108,
    FULL_HOUSE: 3_744,
    QUADS: 624,
    STRAIGHT_FLUSH: 40,
}
# Known counts of 7-card hands by the category of their best 5 cards.
SEVEN_CARD_COUNTS = {
    HIGH_CARD: 23_294_460,
    PAIR: 58_627_800,
    TWO_PAIR: 31_433_400,
    TRIPS: 6_461_620,
    STRAIGHT: 6_180_020,
    FLUSH: 4_047_644,
    FULL_HOUSE: 3_473_184,
    QUADS: 224_848,
    STRAIGHT_FLUSH: 41_584,
}


def strength(text: str) -> int:
    return int(evaluate(np.array(parse_cards(text), dtype=np.int64)))


@njit(cache=True)
def _count_five():
    counts = np.zeros(9, dtype=np.int64)
    values = np.empty(2_598_960, dtype=np.int64)
    hand = np.empty(5, dtype=np.int64)
    n = 0
    for a in range(52):
        hand[0] = a
        for b in range(a + 1, 52):
            hand[1] = b
            for c in range(b + 1, 52):
                hand[2] = c
                for d in range(c + 1, 52):
                    hand[3] = d
                    for e in range(d + 1, 52):
                        hand[4] = e
                        value = evaluate(hand)
                        counts[value >> 20] += 1
                        values[n] = value
                        n += 1
    return counts, values


@njit(cache=True)
def _count_seven():
    counts = np.zeros(9, dtype=np.int64)
    hand = np.empty(7, dtype=np.int64)
    for a in range(52):
        hand[0] = a
        for b in range(a + 1, 52):
            hand[1] = b
            for c in range(b + 1, 52):
                hand[2] = c
                for d in range(c + 1, 52):
                    hand[3] = d
                    for e in range(d + 1, 52):
                        hand[4] = e
                        for f in range(e + 1, 52):
                            hand[5] = f
                            for g in range(f + 1, 52):
                                hand[6] = g
                                counts[evaluate(hand) >> 20] += 1
    return counts


def test_five_card_category_counts_and_distinct_values():
    counts, values = _count_five()
    assert {k: int(counts[k]) for k in range(9)} == FIVE_CARD_COUNTS
    # There are exactly 7,462 distinguishable 5-card hands.
    assert len(np.unique(values)) == 7462


@pytest.mark.slow
def test_seven_card_category_counts():
    counts = _count_seven()
    assert {k: int(counts[k]) for k in range(9)} == SEVEN_CARD_COUNTS


def test_category_order():
    ladder = [
        "AhKd9c7s3h",  # high card
        "2h2d5c7s9h",  # pair
        "2h2d3c3s4h",  # two pair
        "2h2d2c5s7h",  # trips
        "Ah2d3c4s5h",  # wheel, the lowest straight
        "2h4h6h8hTh",  # flush
        "2h2d2c3s3h",  # full house
        "2h2d2c2s3h",  # quads
        "Ah2h3h4h5h",  # steel wheel, the lowest straight flush
    ]
    values = [strength(h) for h in ladder]
    assert values == sorted(values)
    assert [v >> CATEGORY_SHIFT for v in values] == list(range(9))


@pytest.mark.parametrize(
    "better, worse",
    [
        # kickers
        ("AhKd9c7s3h", "AhKd9c7s2h"),
        ("AhAdKc7s3h", "AhAdQc7s3h"),
        ("AhAdKcQs3h", "AhAdKcJs3h"),
        ("AhAdKcKs3h", "AhAdKcKs2h"),
        ("AhAdAcKs3h", "AhAdAcQsJh"),
        ("AhAdAcAsKh", "AhAdAcAsQh"),
        # pair rank beats kickers
        ("3h3d4c5s7h", "2h2dAcKsQh"),
        # two pair: the top pair decides first
        ("AhAd2c2s3h", "KhKdQcQsAh"),
        # straights: six-high beats the wheel, ace-high beats king-high
        ("2h3d4c5s6h", "Ah2d3c4s5h"),
        ("AhKdQcJsTh", "KhQdJcTs9h"),
        # flush ranks
        ("AhKh9h7h3h", "AdQdJdTd8d"),
        # full house: trips rank first
        ("3h3d3c2s2h", "2h2d2cAsAh"),
        # 7 cards: three pairs, the third pair can be the kicker
        ("AhAdKcKsQhQd2c", "AhAdKcKsJhJd2c"),
        # 7 cards: two sets make a full house using the higher set
        ("AhAdAcKsKhKd2c", "KhKdKcQsQhQd2c"),
        # 7 cards: flush beats the straight that is also present
        ("2h4h6h8hTh9c7d", "6h7d8c9sTdJhQc"),
        # 7 cards: straight flush hidden among a higher plain straight
        ("5h6h7h8h9hTcJd", "AhKhQhJh9h2c3d"),
    ],
)
def test_ordering_within_and_across_categories(better, worse):
    assert strength(better) > strength(worse)


def test_board_plays_is_a_tie():
    board = "AhKdQcJsTh"
    assert strength(board + "2c3d") == strength(board + "4c5d")


def test_wheel_does_not_use_ace_as_high():
    assert strength("Ah2d3c4s5h") < strength("2h3d4c5s6h")
    assert category(strength("QhKdAc2s3h")) == HIGH_CARD  # no wrap-around straight


def test_invariant_to_card_order_and_suit_relabelling():
    rng = random.Random(7)
    for _ in range(2000):
        cards = rng.sample(range(N_CARDS), 7)
        base = int(evaluate(np.array(cards, dtype=np.int64)))
        shuffled = cards[:]
        rng.shuffle(shuffled)
        assert int(evaluate(np.array(shuffled, dtype=np.int64))) == base
        perm = [0, 1, 2, 3]
        rng.shuffle(perm)
        relabelled = [(c >> 2) * 4 + perm[c & 3] for c in cards]
        assert int(evaluate(np.array(relabelled, dtype=np.int64))) == base


def test_seven_card_value_is_best_five_card_subset():
    rng = random.Random(11)
    for _ in range(3000):
        cards = rng.sample(range(N_CARDS), 7)
        best = max(
            int(evaluate(np.array(sub, dtype=np.int64)))
            for sub in itertools.combinations(cards, 5)
        )
        assert int(evaluate(np.array(cards, dtype=np.int64))) == best


def test_agrees_with_phevaluator():
    phe = pytest.importorskip("phevaluator")
    rng = random.Random(3)
    for _ in range(20_000):
        cards = rng.sample(range(N_CARDS), 14)
        a, b = cards[:7], cards[7:]
        ours = int(evaluate(np.array(a, dtype=np.int64))) - int(
            evaluate(np.array(b, dtype=np.int64))
        )
        # phevaluator uses the same card numbering; lower rank is stronger.
        theirs = phe.evaluate_cards(*b) - phe.evaluate_cards(*a)
        assert (ours > 0) == (theirs > 0) and (ours == 0) == (theirs == 0)


def test_combo_strengths_marks_blocked_combos():
    board = parse_cards("AhKdQc7s2h")
    values = combo_strengths(board)
    assert (values == -1).sum() == 1326 - 47 * 46 // 2
    assert values.max() >> CATEGORY_SHIFT == STRAIGHT  # JT makes broadway
