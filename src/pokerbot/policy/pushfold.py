"""The heads-up push-or-fold game, solved exactly.

This is a restricted game: the small blind may only go all-in or fold, and
the big blind may only call or fold. It is not the full game. A real small
blind can also limp or make a small raise, so the solution is an equilibrium
of the restricted game only, and it is used only in that game's own
situations at short stacks.

The game is small enough to solve to high accuracy: 169 hand classes per
player, two decisions, three endings. Payoffs are in big blinds.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from pokerbot.cards import CLASS_LABELS, CLASS_SIZE, COMBO_CARDS, COMBO_CLASS, N_CLASSES
from pokerbot.preflop_table import preflop_equity
from pokerbot.solver.public_tree import DenseTerminal, Node, Tree
from pokerbot.solver.vector_cfr import VectorCFR


@lru_cache(maxsize=1)
def class_deal_probability() -> np.ndarray:
    """``p[i, j]``: probability that the small blind is dealt class i and the
    big blind class j. Pairs of combos that share a card cannot be dealt."""
    a, b = COMBO_CARDS[:, 0], COMBO_CARDS[:, 1]
    shares = (
        (a[:, None] == a[None, :])
        | (a[:, None] == b[None, :])
        | (b[:, None] == a[None, :])
        | (b[:, None] == b[None, :])
    )
    compatible = (~shares).astype(np.float64)
    onehot = np.zeros((len(COMBO_CLASS), N_CLASSES))
    onehot[np.arange(len(COMBO_CLASS)), COMBO_CLASS] = 1.0
    counts = onehot.T @ compatible @ onehot
    return counts / counts.sum()


@dataclass
class PushFold:
    stack_bb: float
    jam: np.ndarray  # (169,) probability the small blind goes all-in, by class
    call: np.ndarray  # (169,) probability the big blind calls, by class
    value_sb: float  # small blind's value in big blinds per hand
    exploitability: float  # big blinds per hand, within the restricted game
    iterations: int

    @property
    def jam_frequency(self) -> float:
        return float(self.jam @ CLASS_SIZE / CLASS_SIZE.sum())

    @property
    def call_frequency(self) -> float:
        return float(self.call @ CLASS_SIZE / CLASS_SIZE.sum())

    def chart(self, which: str = "jam") -> str:
        """A 13 x 13 text grid: pairs on the diagonal, suited hands above it."""
        probs = self.jam if which == "jam" else self.call
        by_label = dict(zip(CLASS_LABELS, probs))
        ranks = "AKQJT98765432"
        lines = []
        for i, high in enumerate(ranks):
            cells = []
            for j, low in enumerate(ranks):
                if i == j:
                    label = high + low
                elif i < j:
                    label = high + low + "s"
                else:
                    label = low + high + "o"
                p = by_label[label]
                mark = "#" if p > 0.95 else ("." if p < 0.05 else "+")
                cells.append(f"{label:>4}{mark}")
            lines.append(" ".join(cells))
        return "\n".join(lines)


def pushfold_tree(stack_bb: float, small_blind: float = 0.5, big_blind: float = 1.0) -> Tree:
    deal = class_deal_probability()
    equity = preflop_equity()
    root = Node.decision(
        0,
        ["fold", "jam"],
        [
            Node.leaf(DenseTerminal(deal * -small_blind)),
            Node.decision(
                1,
                ["fold", "call"],
                [
                    Node.leaf(DenseTerminal(deal * big_blind)),
                    Node.leaf(DenseTerminal(deal * stack_bb * (2.0 * equity - 1.0))),
                ],
            ),
        ],
    )
    return Tree(root, (N_CLASSES, N_CLASSES))


@lru_cache(maxsize=64)
def solve_pushfold(stack_bb: float, iterations: int = 3000) -> PushFold:
    """Solve the push-or-fold game for an effective stack in big blinds."""
    if stack_bb <= 1.0:
        raise ValueError("the effective stack must exceed one big blind")
    tree = pushfold_tree(stack_bb)
    solver = VectorCFR(tree, "dcfr")
    solver.iterate(iterations)
    root = tree.root
    return PushFold(
        stack_bb=stack_bb,
        jam=solver.average_strategy(root)[:, 1].copy(),
        call=solver.average_strategy(root.children[1])[:, 1].copy(),
        value_sb=solver.game_value(0),
        exploitability=solver.exploitability(),
        iterations=solver.t,
    )
