"""A game as a tree of public events, with private hands kept as vectors.

Every node is something both players can see: a decision by one player, a
public chance event, or the end of the hand. Each player holds one of a
fixed list of private hands. A strategy at a decision node is a matrix of
hands by actions, and values are vectors with one entry per hand.

Kuhn poker, Leduc poker and a No-Limit Hold'em river all fit this shape, so
the solver that is checked on the two small games is the same code that
solves the river.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DECISION, CHANCE, TERMINAL = "decision", "chance", "terminal"


class Terminal:
    """The end of a hand. ``utility`` returns, for every hand of ``player``,
    the payoff summed over the opponent's hands weighted by ``reach_opp``."""

    def utility(self, player: int, reach_opp: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DenseTerminal(Terminal):
    """Payoffs given as a full matrix.

    ``u0[i, j]`` is player 0's payoff when it holds hand i and the opponent
    holds hand j, already multiplied by the probability of that deal (zero
    for deals that cannot happen).
    """

    def __init__(self, u0: np.ndarray):
        self.u0 = np.ascontiguousarray(u0, dtype=np.float64)

    def utility(self, player: int, reach_opp: np.ndarray) -> np.ndarray:
        if player == 0:
            return self.u0 @ reach_opp
        return -(self.u0.T @ reach_opp)


@dataclass
class Node:
    kind: str
    player: int = -1
    actions: list = field(default_factory=list)  # labels, one per child
    children: list[Node] = field(default_factory=list)
    terminal: Terminal | None = None
    index: int = -1  # position among the decision nodes, set by Tree

    @staticmethod
    def decision(player: int, actions: list, children: list[Node]) -> Node:
        return Node(DECISION, player, list(actions), list(children))

    @staticmethod
    def chance(children: list[Node], labels: list | None = None) -> Node:
        return Node(CHANCE, actions=list(labels or range(len(children))), children=list(children))

    @staticmethod
    def leaf(terminal: Terminal) -> Node:
        return Node(TERMINAL, terminal=terminal)


class Tree:
    def __init__(self, root: Node, n_hands: tuple[int, int]):
        self.root = root
        self.n_hands = n_hands
        self.decisions: list[Node] = []
        self.n_terminals = 0
        stack = [root]
        while stack:
            node = stack.pop()
            if node.kind == DECISION:
                node.index = len(self.decisions)
                self.decisions.append(node)
            elif node.kind == TERMINAL:
                self.n_terminals += 1
            stack.extend(reversed(node.children))
