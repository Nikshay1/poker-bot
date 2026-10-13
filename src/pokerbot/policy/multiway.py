"""Three or more players: equity analysis and a conservative rule.

This is not a solved strategy. With more than two players there is no
equilibrium guarantee to aim for, so this layer reports the hand's equity
against the other players' ranges and applies a cautious rule on top. It
does not model players still to act behind, bluffing, or side pots.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from pokerbot.cards import CLASS_SIZE, COMBO_CLASS, N_COMBOS, cards_to_str
from pokerbot.equity import monte_carlo_equity
from pokerbot.game.state import CALL, CHECK, FOLD, RAISE, Action, SpotError
from pokerbot.preflop_table import equity_vs_random

LAYER_MULTIWAY = "multiway equity analysis"
BETTOR_RANGE_FRAC = 0.30  # a player who bet is assumed to hold a top-30% starting hand


def top_fraction_range(frac: float) -> np.ndarray:
    """The best ``frac`` of starting hands by equity against a random hand."""
    score = equity_vs_random()
    order = np.argsort(-score, kind="stable")
    share = np.cumsum(CLASS_SIZE[order]) / N_COMBOS
    keep = np.zeros(169, dtype=bool)
    keep[order[: int(np.searchsorted(share, frac)) + 1]] = True
    return keep[COMBO_CLASS].astype(np.float64)


@dataclass
class MultiwayRecommendation:
    action: Action
    equity: float
    stderr: float
    fair_share: float
    ev: dict[str, float | None]
    layer: str = LAYER_MULTIWAY
    assumptions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "action": self.action.kind,
            "amount": self.action.amount,
            "equity": self.equity,
            "equity_stderr": self.stderr,
            "fair_share": self.fair_share,
            "ev": self.ev,
            "layer": self.layer,
            "assumptions": self.assumptions,
            "notes": self.notes,
            "seconds": round(self.seconds, 4),
        }

    def render_multiway(self, hole, board, pot, to_call) -> str:
        where = f" on {cards_to_str(board)}" if board else ""
        lines = [
            f"Hand {cards_to_str(hole)}{where}",
            f"Pot {pot}, to call {to_call}",
            "",
            f"Recommended: {self.action}",
            f"Equity: {self.equity:.1%} (±{1.96 * self.stderr:.1%}); "
            f"an equal share would be {self.fair_share:.1%}",
        ]
        evs = [f"{name} {value:+.0f}" for name, value in self.ev.items() if value is not None]
        if evs:
            lines.append("EV in chips, relative to folding: " + ", ".join(evs))
        lines.append(f"Answered by: {self.layer} in {self.seconds * 1000:.0f} ms")
        for title, items in (("Assumptions", self.assumptions), ("Notes", self.notes)):
            lines.append(f"{title}:")
            lines.extend(f"  - {item}" for item in items)
        return "\n".join(lines)


def advise_multiway(
    hole,
    board=(),
    pot: int | None = None,
    to_call: int = 0,
    stack: int | None = None,
    n_opponents: int = 2,
    big_blind: int = 100,
    n_samples: int = 40_000,
    seed: int = 0,
) -> MultiwayRecommendation:
    started = time.perf_counter()
    hole, board = tuple(hole), tuple(board)
    if n_opponents < 2:
        raise SpotError("not_multiway", "use the heads-up advisor for one opponent")
    if n_opponents > 8:
        raise SpotError("too_many_players", "at most nine players")
    if pot is None or pot <= 0:
        raise SpotError("missing_pot", "the pot size is required")
    if to_call < 0 or to_call >= pot:
        raise SpotError("bad_to_call", "the amount to call must be less than the pot")
    if len(set(hole) | set(board)) != len(hole) + len(board):
        raise SpotError("duplicate_card", "the same card appears twice")
    if stack is not None and to_call > stack:
        raise SpotError(
            "side_pot_unsupported",
            "calling for less than the bet creates a side pot, which this layer does not model",
        )

    assumptions = [f"{n_opponents} opponents, all still in the hand", "no rake"]
    ranges = [np.ones(N_COMBOS) for _ in range(n_opponents)]
    if to_call > 0:
        ranges[0] = top_fraction_range(BETTOR_RANGE_FRAC)
        assumptions.append(
            f"the player who bet holds a top-{BETTOR_RANGE_FRAC:.0%} starting hand; "
            "the others hold random hands"
        )
    else:
        assumptions.append("every opponent holds a random hand")

    result = monte_carlo_equity(hole, board, ranges, n_samples=n_samples, seed=seed)
    equity, stderr = result["equity"], result["stderr"]
    fair = 1.0 / (n_opponents + 1)
    notes = [
        "equity counts split pots as fractional shares",
        "players still to act and later betting are not modelled",
    ]
    ev: dict[str, float | None] = {}

    # Extra opponents can still raise behind, so ask for more than bare pot odds.
    margin = 0.03 * (n_opponents - 1)
    strong = equity >= min(2.0 * fair, 0.85)
    if to_call == 0:
        if strong:
            amount = int(round(0.66 * pot))
            if stack is not None:
                amount = min(amount, stack)
            action = Action(RAISE, max(amount, big_blind))
            notes.append("bets because equity is at least twice an equal share")
        else:
            action = Action(CHECK)
    else:
        need = to_call / (pot + to_call)
        ev[FOLD] = 0.0
        ev[CALL] = equity * (pot + to_call) - to_call
        notes.append(f"needs equity of {need:.1%} to call, plus a margin of {margin:.0%}")
        if strong and (stack is None or stack > to_call):
            amount = 3 * to_call if stack is None else min(3 * to_call, stack)
            action = Action(RAISE, amount)
            notes.append("raises because equity is at least twice an equal share")
        elif equity >= need + margin:
            action = Action(CALL, to_call)
        else:
            action = Action(FOLD)

    return MultiwayRecommendation(
        action=action,
        equity=equity,
        stderr=stderr,
        fair_share=fair,
        ev=ev,
        assumptions=assumptions,
        notes=notes,
        seconds=time.perf_counter() - started,
    )
