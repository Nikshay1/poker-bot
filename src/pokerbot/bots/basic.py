"""Bots: the interface, three simple baselines and the heuristic bot.

A bot only ever receives a ``Spot``, which holds its own cards and the public
state. It never sees the engine or the opponent's cards.
"""

from __future__ import annotations

import numpy as np

from pokerbot.advisor import SolverConfig, advise
from pokerbot.cards import COMBO_CLASS, combo_index
from pokerbot.equity import range_equity, uniform_range
from pokerbot.game.state import CALL, CHECK, FOLD, RAISE, Action, Spot
from pokerbot.policy.base import PolicyConfig
from pokerbot.preflop_table import equity_vs_random
from pokerbot.ranges import RangeTracker


class Bot:
    name = "bot"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def new_hand(self) -> None:
        """Called before each hand."""

    def act(self, spot: Spot) -> Action:
        raise NotImplementedError


class RandomBot(Bot):
    """Picks uniformly among fold, check/call and a random raise."""

    name = "random"

    def act(self, spot: Spot) -> Action:
        legal = spot.legal()
        options = [Action(CHECK) if legal.can_check else Action(CALL, legal.call_amount)]
        if legal.can_fold:
            options.append(Action(FOLD))
        if legal.can_raise:
            top = min(legal.max_raise_to, legal.min_raise_to + spot.pot)
            options.append(Action(RAISE, int(self.rng.integers(legal.min_raise_to, top + 1))))
        return options[int(self.rng.integers(len(options)))]


class CallBot(Bot):
    """Never folds, never raises."""

    name = "call"

    def act(self, spot: Spot) -> Action:
        legal = spot.legal()
        return Action(CHECK) if legal.can_check else Action(CALL, legal.call_amount)


class EquityBot(Bot):
    """Plays its equity against a random hand: raise when far ahead, call when
    pot odds are met, otherwise check or fold. It ignores what the betting
    says about the opponent's hand."""

    name = "equity"

    def act(self, spot: Spot) -> Action:
        legal = spot.legal()
        combo = combo_index(*spot.hole)
        if spot.street == 0:
            equity = float(equity_vs_random()[COMBO_CLASS[combo]])
        else:
            equity = float(range_equity(spot.board, uniform_range())[combo])
        pot, to_call = spot.pot, legal.call_amount
        if to_call == 0:
            if equity > 0.65 and legal.can_raise:
                target = spot.street_contrib[spot.seat] + 0.66 * pot
                return Action(RAISE, legal.clamp_raise(int(target)))
            return Action(CHECK)
        if equity > 0.75 and legal.can_raise:
            return Action(RAISE, legal.clamp_raise(3 * spot.street_contrib[1 - spot.seat]))
        if equity >= to_call / (pot + to_call):
            return Action(CALL, to_call)
        return Action(FOLD)


class HeuristicBot(Bot):
    """The heuristic advisor with range tracking, sampling its mixed strategy."""

    name = "heuristic"

    def __init__(self, seed: int = 0, cfg: PolicyConfig | None = None):
        super().__init__(seed)
        self.cfg = cfg or PolicyConfig()
        self.tracker = RangeTracker(self.cfg)

    def new_hand(self) -> None:
        self.tracker = RangeTracker(self.cfg)

    def act(self, spot: Spot) -> Action:
        return advise(spot, self.cfg, self.tracker, self.rng).action


class SolverBot(HeuristicBot):
    """The heuristic bot with river decisions solved by CFR."""

    name = "solver"
    solver = SolverConfig()

    def act(self, spot: Spot) -> Action:
        return advise(spot, self.cfg, self.tracker, self.rng, self.solver).action


BOTS = {cls.name: cls for cls in (RandomBot, CallBot, EquityBot, HeuristicBot, SolverBot)}


def make_bot(name: str, seed: int = 0) -> Bot:
    if name not in BOTS:
        raise ValueError(f"unknown bot {name!r}; choose from {sorted(BOTS)}")
    return BOTS[name](seed)
