"""Equity: the share of the pot a hand wins at showdown, ties counted as halves.

``range_equity`` is the workhorse. For a board and an opponent range it
returns the equity of every hole-card combo at once, exactly, with combos
that share a card excluded.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from pokerbot.cards import COMBO_CARDS, N_COMBOS
from pokerbot.evaluator import evaluate


@njit(cache=True)
def showdown_sums(strengths, opp_weights, combo_cards, win, tie, total):
    """Add to ``win``, ``tie`` and ``total`` the opponent weight each combo
    beats, ties and faces on one complete board.

    A combo only faces opponent combos that share no card with it. Sorting by
    strength makes "weight of all weaker combos" a running sum; the combos
    that share a card are then removed with per-card running sums. The combo
    itself is subtracted once for each of its two cards, so it is added back.
    Combos with negative strength (blocked by the board) are skipped.
    """
    n = strengths.shape[0]
    order = np.argsort(strengths)
    card_all = np.zeros(52)
    card_less = np.zeros(52)
    card_equal = np.zeros(52)
    weight_all = 0.0
    for k in range(n):
        w = opp_weights[k]
        if strengths[k] >= 0 and w > 0.0:
            weight_all += w
            card_all[combo_cards[k, 0]] += w
            card_all[combo_cards[k, 1]] += w

    i = 0
    while i < n and strengths[order[i]] < 0:
        i += 1
    weight_less = 0.0
    while i < n:
        strength = strengths[order[i]]
        j = i
        weight_equal = 0.0
        while j < n and strengths[order[j]] == strength:
            k = order[j]
            w = opp_weights[k]
            if w > 0.0:
                weight_equal += w
                card_equal[combo_cards[k, 0]] += w
                card_equal[combo_cards[k, 1]] += w
            j += 1
        for t in range(i, j):
            k = order[t]
            a = combo_cards[k, 0]
            b = combo_cards[k, 1]
            w = opp_weights[k]
            win[k] += weight_less - card_less[a] - card_less[b]
            tie[k] += weight_equal - card_equal[a] - card_equal[b] + w
            total[k] += weight_all - card_all[a] - card_all[b] + w
        for t in range(i, j):
            k = order[t]
            w = opp_weights[k]
            a = combo_cards[k, 0]
            b = combo_cards[k, 1]
            if w > 0.0:
                card_less[a] += w
                card_less[b] += w
            card_equal[a] = 0.0
            card_equal[b] = 0.0
        weight_less += weight_equal
        i = j


@njit(cache=True)
def _strengths_into(cards, n_cards, dead, combo_cards, hand, out):
    for i in range(n_cards):
        hand[i] = cards[i]
    for k in range(combo_cards.shape[0]):
        c1 = combo_cards[k, 0]
        c2 = combo_cards[k, 1]
        if dead[c1] or dead[c2]:
            out[k] = -1
        else:
            hand[n_cards] = c1
            hand[n_cards + 1] = c2
            out[k] = evaluate(hand)


@njit(cache=True, parallel=True)
def _range_sums(board, opp_weights, combo_cards):
    """Win, tie and total weight for every combo, summed over all runouts."""
    n_board = board.shape[0]
    n = combo_cards.shape[0]
    # One slot per first runout card keeps the parallel loop free of shared writes.
    win = np.zeros((52, n))
    tie = np.zeros((52, n))
    total = np.zeros((52, n))
    on_board = np.zeros(52, dtype=np.bool_)
    for i in range(n_board):
        on_board[board[i]] = True

    if n_board == 5:
        dead = on_board.copy()
        strengths = np.empty(n, dtype=np.int64)
        hand = np.empty(7, dtype=np.int64)
        _strengths_into(board, 5, dead, combo_cards, hand, strengths)
        showdown_sums(strengths, opp_weights, combo_cards, win[0], tie[0], total[0])
    else:
        for first in prange(52):
            if on_board[first]:
                continue
            cards = np.empty(5, dtype=np.int64)
            for i in range(n_board):
                cards[i] = board[i]
            cards[n_board] = first
            dead = on_board.copy()
            dead[first] = True
            strengths = np.empty(n, dtype=np.int64)
            hand = np.empty(7, dtype=np.int64)
            if n_board == 4:
                _strengths_into(cards, 5, dead, combo_cards, hand, strengths)
                showdown_sums(
                    strengths, opp_weights, combo_cards, win[first], tie[first], total[first]
                )
            else:
                for second in range(first + 1, 52):
                    if on_board[second]:
                        continue
                    cards[4] = second
                    dead[second] = True
                    _strengths_into(cards, 5, dead, combo_cards, hand, strengths)
                    showdown_sums(
                        strengths, opp_weights, combo_cards, win[first], tie[first], total[first]
                    )
                    dead[second] = False
    return win.sum(axis=0), tie.sum(axis=0), total.sum(axis=0)


def range_sums(board, opp_weights):
    """Raw win, tie and total weights behind ``range_equity``."""
    board = np.asarray(board, dtype=np.int64)
    if not 3 <= len(board) <= 5:
        raise ValueError("range_equity needs a flop, turn or river board")
    weights = np.ascontiguousarray(opp_weights, dtype=np.float64)
    if weights.shape != (N_COMBOS,):
        raise ValueError("a range is a vector of 1326 weights")
    return _range_sums(board, weights, COMBO_CARDS)


def range_equity(board, opp_weights) -> np.ndarray:
    """Exact equity of every combo against an opponent range on a 3 to 5 card
    board. Combos that are blocked, or face no compatible opponent combo, get NaN.

    Every (runout, opponent combo) pair that is compatible with the hero's
    combo is equally likely up to the opponent's weight, so summing wins over
    all such pairs and dividing by their total weight is exact.
    """
    win, tie, total = range_sums(board, opp_weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (win + 0.5 * tie) / total, np.nan)


@njit(cache=True)
def _hand_vs_hand(hole_a, hole_b, board):
    """Exact win/tie/loss counts for hand a against hand b over all runouts."""
    n_board = board.shape[0]
    need = 5 - n_board
    dead = np.zeros(52, dtype=np.bool_)
    for c in hole_a:
        dead[c] = True
    for c in hole_b:
        dead[c] = True
    for c in board:
        dead[c] = True
    deck = np.empty(52, dtype=np.int64)
    n_deck = 0
    for c in range(52):
        if not dead[c]:
            deck[n_deck] = c
            n_deck += 1

    a = np.empty(7, dtype=np.int64)
    b = np.empty(7, dtype=np.int64)
    a[0], a[1] = hole_a[0], hole_a[1]
    b[0], b[1] = hole_b[0], hole_b[1]
    for i in range(n_board):
        a[2 + i] = board[i]
        b[2 + i] = board[i]

    counts = np.zeros(3, dtype=np.int64)
    idx = np.arange(need)
    while True:
        for i in range(need):
            a[2 + n_board + i] = deck[idx[i]]
            b[2 + n_board + i] = deck[idx[i]]
        sa = evaluate(a)
        sb = evaluate(b)
        if sa > sb:
            counts[0] += 1
        elif sa == sb:
            counts[1] += 1
        else:
            counts[2] += 1
        # advance to the next combination of deck positions
        i = need - 1
        while i >= 0 and idx[i] == n_deck - need + i:
            i -= 1
        if i < 0:
            break
        idx[i] += 1
        for j in range(i + 1, need):
            idx[j] = idx[j - 1] + 1
    return counts


def hand_vs_hand(hole_a, hole_b, board=()) -> tuple[float, float, float]:
    """Exact (win, tie, loss) probabilities for hand a against hand b."""
    cards = list(hole_a) + list(hole_b) + list(board)
    if len(set(cards)) != len(cards):
        raise ValueError("hands and board share a card")
    counts = _hand_vs_hand(
        np.asarray(hole_a, dtype=np.int64),
        np.asarray(hole_b, dtype=np.int64),
        np.asarray(board, dtype=np.int64),
    )
    n = counts.sum()
    return counts[0] / n, counts[1] / n, counts[2] / n


@njit(cache=True)
def _monte_carlo(hole, board, cum_weights, combo_cards, n_samples, seed):
    """Hero's pot share against several opponents, each drawn from its range.

    Opponent hands are drawn jointly: a draw in which any two hands collide is
    thrown away whole, which leaves exactly the distribution proportional to
    the product of the range weights over non-colliding deals.
    """
    np.random.seed(seed)
    n_opp = cum_weights.shape[0]
    n_board = board.shape[0]
    need = 5 - n_board
    base_dead = np.zeros(52, dtype=np.bool_)
    base_dead[hole[0]] = True
    base_dead[hole[1]] = True
    for c in board:
        base_dead[c] = True

    hand = np.empty(7, dtype=np.int64)
    full_board = np.empty(5, dtype=np.int64)
    for i in range(n_board):
        full_board[i] = board[i]
    opp = np.empty((n_opp, 2), dtype=np.int64)
    deck = np.empty(52, dtype=np.int64)

    share_sum = 0.0
    share_sq = 0.0
    wins = 0
    ties = 0
    done = 0
    attempts = 0
    max_attempts = 200 * n_samples + 1000
    while done < n_samples and attempts < max_attempts:
        attempts += 1
        dead = base_dead.copy()
        ok = True
        for p in range(n_opp):
            k = np.searchsorted(cum_weights[p], np.random.random() * cum_weights[p, -1], side="right")
            if k >= combo_cards.shape[0]:
                k = combo_cards.shape[0] - 1
            c1 = combo_cards[k, 0]
            c2 = combo_cards[k, 1]
            if dead[c1] or dead[c2]:
                ok = False
                break
            dead[c1] = True
            dead[c2] = True
            opp[p, 0] = c1
            opp[p, 1] = c2
        if not ok:
            continue

        n_deck = 0
        for c in range(52):
            if not dead[c]:
                deck[n_deck] = c
                n_deck += 1
        for i in range(need):
            j = i + int(np.random.random() * (n_deck - i))
            deck[i], deck[j] = deck[j], deck[i]
            full_board[n_board + i] = deck[i]

        for i in range(5):
            hand[i] = full_board[i]
        hand[5] = hole[0]
        hand[6] = hole[1]
        hero = evaluate(hand)
        best = hero
        n_best = 1
        for p in range(n_opp):
            hand[5] = opp[p, 0]
            hand[6] = opp[p, 1]
            s = evaluate(hand)
            if s > best:
                best = s
                n_best = 1
            elif s == best:
                n_best += 1
        share = 0.0
        if hero == best:
            share = 1.0 / n_best
            if n_best == 1:
                wins += 1
            else:
                ties += 1
        share_sum += share
        share_sq += share * share
        done += 1
    return share_sum, share_sq, wins, ties, done


def monte_carlo_equity(hole, board, opp_ranges, n_samples=20_000, seed=0) -> dict:
    """Hero's expected pot share against one or more opponent ranges.

    ``opp_ranges`` is a list of 1326-weight vectors, one per opponent. Works
    on any street including before the flop. Returns the estimate with its
    standard error.
    """
    hole = np.asarray(hole, dtype=np.int64)
    board = np.asarray(board, dtype=np.int64)
    dead = np.zeros(52, dtype=bool)
    dead[hole] = True
    dead[board] = True
    blocked = dead[COMBO_CARDS[:, 0]] | dead[COMBO_CARDS[:, 1]]
    cum = np.empty((len(opp_ranges), N_COMBOS))
    for p, weights in enumerate(opp_ranges):
        w = np.where(blocked, 0.0, np.asarray(weights, dtype=np.float64))
        if w.sum() <= 0:
            raise ValueError("an opponent range has no hand compatible with the known cards")
        cum[p] = np.cumsum(w)
    share_sum, share_sq, wins, ties, done = _monte_carlo(
        hole, board, cum, COMBO_CARDS, int(n_samples), int(seed)
    )
    if done == 0:
        raise ValueError("could not sample a compatible deal from these ranges")
    mean = share_sum / done
    var = max(share_sq / done - mean * mean, 0.0)
    return {
        "equity": mean,
        "stderr": (var / done) ** 0.5,
        "win": wins / done,
        "tie": ties / done,
        "samples": done,
    }


def uniform_range() -> np.ndarray:
    return np.ones(N_COMBOS)
