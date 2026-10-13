"""Solve a heads-up river decision.

Inputs are the situation, both players' ranges and the bet sizes to keep in
the tree. The output is the equilibrium strategy of that game for every hand
of the player to act, with the value of each action.

What the result means, and what it does not:

* "Exploitability" is exact for this bet tree and these starting ranges.
  A real opponent may use sizes outside the tree or hold a different range,
  and the number says nothing about that.
* The solve runs in small batches and can be stopped by a time budget. It
  always returns a valid strategy and reports how it stopped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from pokerbot.cards import combo_index
from pokerbot.game.state import Action, Spot
from pokerbot.game.tree import BetSizing, BetState, tree_actions
from pokerbot.solver.public_tree import Node, Tree
from pokerbot.solver.terminal import FoldTerminal, RiverBoard, ShowdownTerminal
from pokerbot.solver.vector_cfr import VectorCFR


@dataclass
class RiverSolution:
    seat: int  # the player to act at the root
    actions: list[Action]
    strategy: np.ndarray  # (1326, actions): the root player's average strategy
    action_values: (
        np.ndarray
    )  # (1326, actions): net chips from the start of the hand; NaN if impossible
    fold_value: float  # what folding now is worth, on the same scale
    pot: int
    exploitability: float | None  # chips, within this tree and these ranges
    iterations: int
    seconds: float
    stopped: str  # "converged", "time" or "iterations"
    n_decisions: int
    n_terminals: int

    @property
    def exploitability_pct_pot(self) -> float | None:
        return None if self.exploitability is None else 100.0 * self.exploitability / self.pot

    def for_hand(self, hole) -> tuple[np.ndarray, np.ndarray]:
        """Strategy and action values (relative to folding now) for one hand."""
        k = combo_index(*hole)
        return self.strategy[k], self.action_values[k] - self.fold_value


def build_river_tree(state: BetState, board: RiverBoard, sizing: BetSizing) -> Tree:
    def build(s: BetState) -> Node:
        if s.folded is not None:
            return Node.leaf(FoldTerminal(board, s.folded, s.total_contrib[s.folded]))
        if s.closed:
            if s.total_contrib[0] != s.total_contrib[1]:
                raise ValueError("a showdown needs matched contributions")
            return Node.leaf(ShowdownTerminal(board, s.total_contrib[0]))
        actions = tree_actions(s, sizing)
        return Node.decision(s.to_act, actions, [build(s.apply(a)) for a in actions])

    return Tree(build(state), (1326, 1326))


def solve_river(
    spot: Spot,
    ranges: tuple[np.ndarray, np.ndarray],
    sizing: BetSizing | None = None,
    variant: str = "dcfr",
    time_budget: float = 1.0,
    max_iterations: int = 1000,
    target_pct_pot: float | None = 0.5,
    batch: int = 10,
    measure: bool = True,
) -> RiverSolution:
    """Solve the river subgame that starts at ``spot``.

    ``ranges`` are both players' ranges by seat, as they stand when the hero
    is about to act. Stops when exploitability falls to ``target_pct_pot``
    percent of the pot, at ``max_iterations``, or when ``time_budget``
    seconds run out, whichever comes first.
    """
    started = time.perf_counter()
    if len(spot.board) != 5:
        raise ValueError("solve_river needs a five-card board")
    sizing = sizing or BetSizing()
    board = RiverBoard(spot.board)
    reach = []
    for seat in (0, 1):
        r = np.where(board.live, np.asarray(ranges[seat], dtype=np.float64), 0.0)
        if r.sum() <= 0:
            raise ValueError(f"seat {seat} has no possible hand on this board")
        reach.append(r / r.sum())
    norm = float(reach[0] @ board.compatible(reach[1]))
    if norm <= 0:
        raise ValueError("the two ranges cannot both be dealt")

    tree = build_river_tree(BetState.from_spot(spot), board, sizing)
    solver = VectorCFR(tree, variant, (reach[0], reach[1]), norm)
    pot = spot.pot

    exploitability = None
    stopped = "iterations"
    step = min(batch, 5)  # a first short batch, so a strategy exists almost at once
    batches = 0
    while solver.t < max_iterations:
        todo = min(step, max_iterations - solver.t)
        batch_started = time.perf_counter()
        solver.iterate(todo)
        per_iteration = (time.perf_counter() - batch_started) / todo
        batches += 1
        step = batch
        if target_pct_pot is not None and batches % 4 == 0:
            exploitability = solver.exploitability()
            if 100.0 * exploitability / pot <= target_pct_pot:
                stopped = "converged"
                break
            exploitability = None  # stale once more iterations run
        # Stop if another batch would overrun the budget.
        if time.perf_counter() - started + per_iteration * batch > time_budget:
            stopped = "time"
            break
    if measure and exploitability is None:
        exploitability = solver.exploitability()

    seat = spot.seat
    root = tree.root
    opp_compatible = board.compatible(reach[1 - seat])
    values = np.full((1326, len(root.children)), np.nan)
    possible = opp_compatible > 0
    for a, child in enumerate(root.children):
        counterfactual = solver._value(child, seat, reach[1 - seat], False)
        values[possible, a] = counterfactual[possible] / opp_compatible[possible]
    return RiverSolution(
        seat=seat,
        actions=list(root.actions),
        strategy=solver.average_strategy(root),
        action_values=values,
        fold_value=-float(spot.total_contrib[seat]),
        pot=pot,
        exploitability=exploitability,
        iterations=solver.t,
        seconds=time.perf_counter() - started,
        stopped=stopped,
        n_decisions=len(tree.decisions),
        n_terminals=tree.n_terminals,
    )
