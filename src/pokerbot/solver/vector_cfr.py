"""Counterfactual regret minimization in vector form, and exact best response.

Three variants share one traversal:

* ``cfr``  : plain CFR, uniform averaging.
* ``cfr+`` : regrets floored at zero, average weighted by iteration number.
* ``dcfr`` : Discounted CFR (Brown and Sandholm 2019). Before each update,
  positive regrets are multiplied by t^a / (t^a + 1), negative regrets by
  t^b / (t^b + 1) and the strategy sum by (t / (t + 1))^g, with the
  recommended a = 1.5, b = 0, g = 2.

Updates alternate between the players. The average strategy is the one that
converges to equilibrium; the current strategy does not.

Exploitability is measured exactly: for each player, the best response to
the other's average strategy is found by taking the best action for every
hand at every decision. The reported number is half the sum of the two
best-response values (NashConv / 2), which is zero only at an equilibrium.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from pokerbot.solver.public_tree import CHANCE, TERMINAL, Node, Tree

VARIANTS = ("cfr", "cfr+", "dcfr")

# The two kernels below hold all the arithmetic done at a decision node. They
# are compiled because, with a thousand hands and a handful of actions, the
# overhead of separate NumPy calls costs more than the arithmetic itself.


@njit(cache=True)
def _regret_matching(regret, out):
    n, n_actions = regret.shape
    uniform = 1.0 / n_actions
    for i in range(n):
        total = 0.0
        for a in range(n_actions):
            if regret[i, a] > 0.0:
                total += regret[i, a]
        if total > 0.0:
            for a in range(n_actions):
                out[i, a] = regret[i, a] / total if regret[i, a] > 0.0 else 0.0
        else:
            for a in range(n_actions):
                out[i, a] = uniform


@njit(cache=True)
def _update_node(
    regret, strategy_sum, strategy, action_values, reach_self, mode, t, pos, neg, avg, value
):
    """Node value per hand, then the regret and average-strategy updates.

    ``mode`` is 0 for plain CFR, 1 for CFR+ and 2 for Discounted CFR.
    """
    n, n_actions = regret.shape
    for i in range(n):
        v = 0.0
        for a in range(n_actions):
            v += strategy[i, a] * action_values[i, a]
        value[i] = v
        own = reach_self[i]
        for a in range(n_actions):
            instant = action_values[i, a] - v
            played = own * strategy[i, a]
            if mode == 0:
                regret[i, a] += instant
                strategy_sum[i, a] += played
            elif mode == 1:
                r = regret[i, a] + instant
                regret[i, a] = max(0.0, r)
                strategy_sum[i, a] += t * played
            else:
                r = regret[i, a]
                regret[i, a] = (r * pos if r > 0.0 else r * neg) + instant
                strategy_sum[i, a] = strategy_sum[i, a] * avg + played


def regret_matching(regret: np.ndarray) -> np.ndarray:
    """Play each action in proportion to its positive regret; uniform if none."""
    out = np.empty(regret.shape)
    _regret_matching(np.ascontiguousarray(regret, dtype=np.float64), out)
    return out


class VectorCFR:
    def __init__(
        self,
        tree: Tree,
        variant: str = "dcfr",
        initial_reach: tuple[np.ndarray, np.ndarray] | None = None,
        norm: float = 1.0,
        alpha: float = 1.5,
        beta: float = 0.0,
        gamma: float = 2.0,
    ):
        """``initial_reach`` holds each player's prior over its hands when the
        terminals do not already include it; ``norm`` then divides root values
        so that they are expectations."""
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.tree = tree
        self.variant = variant
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        n0, n1 = tree.n_hands
        if initial_reach is None:
            initial_reach = (np.ones(n0), np.ones(n1))
        self.initial_reach = tuple(np.asarray(r, dtype=np.float64) for r in initial_reach)
        self.norm = norm
        self.regret = [
            np.zeros((tree.n_hands[node.player], len(node.children))) for node in tree.decisions
        ]
        self.strategy_sum = [np.zeros_like(r) for r in self.regret]
        self.t = 0
        self._mode = VARIANTS.index(variant)
        self._pos = self._neg = self._avg = 1.0

    # Strategies -----------------------------------------------------------

    def current_strategy(self, node: Node) -> np.ndarray:
        return regret_matching(self.regret[node.index])

    def average_strategy(self, node: Node) -> np.ndarray:
        total = self.strategy_sum[node.index]
        norm = total.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(norm > 0, total / norm, 1.0 / total.shape[1])

    # Training -------------------------------------------------------------

    def iterate(self, iterations: int = 1) -> None:
        for _ in range(iterations):
            self.t += 1
            t = self.t
            if self.variant == "dcfr":
                self._pos = t**self.alpha / (t**self.alpha + 1.0)
                self._neg = t**self.beta / (t**self.beta + 1.0)
                self._avg = (t / (t + 1.0)) ** self.gamma
            for player in (0, 1):
                self._update(
                    self.tree.root,
                    player,
                    self.initial_reach[1 - player],
                    self.initial_reach[player],
                )

    def _update(self, node: Node, player: int, reach_opp, reach_self) -> np.ndarray:
        """Counterfactual values of ``player``'s hands, updating its regrets."""
        if node.kind == TERMINAL:
            return node.terminal.utility(player, reach_opp)
        if node.kind == CHANCE:
            # Outcome probabilities and blocked deals live in the terminals.
            value = self._update(node.children[0], player, reach_opp, reach_self)
            for child in node.children[1:]:
                value = value + self._update(child, player, reach_opp, reach_self)
            return value

        strategy = self.current_strategy(node)
        if node.player != player:
            value = 0.0
            for a, child in enumerate(node.children):
                value = value + self._update(child, player, reach_opp * strategy[:, a], reach_self)
            return value

        action_values = np.empty_like(strategy)
        for a, child in enumerate(node.children):
            action_values[:, a] = self._update(
                child, player, reach_opp, reach_self * strategy[:, a]
            )
        value = np.empty(strategy.shape[0])
        i = node.index
        _update_node(
            self.regret[i],
            self.strategy_sum[i],
            strategy,
            action_values,
            reach_self,
            self._mode,
            float(self.t),
            self._pos,
            self._neg,
            self._avg,
            value,
        )
        return value

    # Evaluation -----------------------------------------------------------

    def _value(self, node: Node, player: int, reach_opp, best_response: bool) -> np.ndarray:
        """Values of ``player``'s hands when the opponent plays its average
        strategy and ``player`` plays its average strategy or a best response."""
        if node.kind == TERMINAL:
            return node.terminal.utility(player, reach_opp)
        if node.kind == CHANCE:
            value = 0.0
            for child in node.children:
                value = value + self._value(child, player, reach_opp, best_response)
            return value
        strategy = self.average_strategy(node)
        if node.player != player:
            value = 0.0
            for a, child in enumerate(node.children):
                value = value + self._value(
                    child, player, reach_opp * strategy[:, a], best_response
                )
            return value
        children = [self._value(c, player, reach_opp, best_response) for c in node.children]
        if best_response:
            # A hand is one information set per public node, so the best
            # action can be chosen hand by hand.
            return np.max(np.stack(children, axis=1), axis=1)
        return (strategy * np.stack(children, axis=1)).sum(axis=1)

    def hand_values(self, player: int, best_response: bool = False) -> np.ndarray:
        """Counterfactual value of every hand of ``player`` at the root."""
        return self._value(self.tree.root, player, self.initial_reach[1 - player], best_response)

    def game_value(self, player: int = 0) -> float:
        """Expected payoff of ``player`` when both play their average strategies."""
        return float(self.initial_reach[player] @ self.hand_values(player)) / self.norm

    def best_response_value(self, player: int) -> float:
        values = self.hand_values(player, best_response=True)
        return float(self.initial_reach[player] @ values) / self.norm

    def exploitability(self) -> float:
        """NashConv / 2: the average amount a best-responding opponent gains."""
        return 0.5 * (self.best_response_value(0) + self.best_response_value(1))
