"""Kuhn and Leduc poker as public trees.

Both have known solutions, which is what makes them useful: they check the
solver before it is trusted on a game whose answer nobody knows.
"""

from __future__ import annotations

import numpy as np

from pokerbot.solver.public_tree import DenseTerminal, Node, Tree

KUHN_VALUE = -1.0 / 18.0  # first player's value at equilibrium


def kuhn_tree() -> Tree:
    """Kuhn poker: three cards (J, Q, K), one each, ante 1, a single bet of 1."""
    deal = (1.0 - np.eye(3)) / 6.0  # the two players cannot hold the same card
    cards = np.arange(3)
    sign = np.sign(cards[:, None] - cards[None, :]).astype(np.float64)

    def showdown(each: float) -> Node:
        return Node.leaf(DenseTerminal(deal * sign * each))

    def fold(folder: int, lost: float) -> Node:
        return Node.leaf(DenseTerminal(deal * (-lost if folder == 0 else lost)))

    root = Node.decision(
        0,
        ["check", "bet"],
        [
            Node.decision(
                1,
                ["check", "bet"],
                [
                    showdown(1.0),
                    Node.decision(0, ["fold", "call"], [fold(0, 1.0), showdown(2.0)]),
                ],
            ),
            Node.decision(1, ["fold", "call"], [fold(1, 1.0), showdown(2.0)]),
        ],
    )
    return Tree(root, (3, 3))


def leduc_tree() -> Tree:
    """Leduc poker: six cards (J, Q, K in two suits), ante 1, two betting
    rounds with bets of 2 then 4 and at most two raises per round. One public
    card is dealt between the rounds; a hand that pairs it wins, otherwise
    the higher rank wins. Player 0 acts first in both rounds.
    """
    n = 6
    rank = np.arange(n) // 2
    deal = (1.0 - np.eye(n)) / (n * (n - 1))
    bet_size = {1: 2.0, 2: 4.0}

    def showdown(weights: np.ndarray, public: int, each: float) -> Node:
        pair = (rank == rank[public]).astype(np.int64)
        strength = pair * 10 + rank
        sign = np.sign(strength[:, None] - strength[None, :]).astype(np.float64)
        return Node.leaf(DenseTerminal(weights * sign * each))

    def fold(weights: np.ndarray, folder: int, lost: float) -> Node:
        return Node.leaf(DenseTerminal(weights * (-lost if folder == 0 else lost)))

    def end_round(weights, public, round_no, each) -> Node:
        if round_no == 2:
            return showdown(weights, public, each)
        children = []
        for card in range(n):
            # The public card is one of the four cards neither player holds.
            free = np.ones((n, n))
            free[card, :] = 0.0
            free[:, card] = 0.0
            children.append(betting(weights * free / (n - 2), card, 2, [each, each], 0, 0))
        return Node.chance(children)

    def betting(weights, public, round_no, contrib, player, raises) -> Node:
        opp = 1 - player
        to_call = contrib[opp] - contrib[player]
        actions, children = [], []
        if to_call > 0:
            actions.append("fold")
            children.append(fold(weights, player, contrib[player]))
            actions.append("call")
            children.append(end_round(weights, public, round_no, contrib[opp]))
        else:
            actions.append("check")
            if player == 1:  # both have checked
                children.append(end_round(weights, public, round_no, contrib[0]))
            else:
                children.append(betting(weights, public, round_no, contrib, 1, raises))
        if raises < 2:
            raised = list(contrib)
            raised[player] = contrib[opp] + bet_size[round_no]
            actions.append("raise")
            children.append(betting(weights, public, round_no, raised, opp, raises + 1))
        return Node.decision(player, actions, children)

    return Tree(betting(deal, -1, 1, [1.0, 1.0], 0, 0), (n, n))
