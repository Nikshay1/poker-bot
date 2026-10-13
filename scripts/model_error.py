"""Model error of the river solver: is it solving the right game?

A solve can be numerically exact and still answer the wrong question,
because the opponent's range and the bet sizes are assumptions. This script
holds a set of river situations fixed and changes one assumption at a time:

* ``soft``    : ranges inferred with a larger uniform component (epsilon 0.3
                instead of 0.1), so past actions narrow them less;
* ``uniform`` : the opponent may hold any hand;
* ``sizes``   : a richer bet tree (three bet sizes, two raise sizes).

For the hero's hand it reports the recommended action under each, and the
total-variation distance between each variant's strategy and the base one
(0 means identical, 1 means completely different). Distances are compared
after grouping actions into fold, check or call, and bet or raise, so that
trees with different sizes can be compared.

    uv run python scripts/model_error.py
"""

import json
import pathlib

import numpy as np

from pokerbot.advisor import spot_from_partial
from pokerbot.cards import combo_index, parse_cards
from pokerbot.game.state import CALL, CHECK, FOLD
from pokerbot.game.tree import BetSizing
from pokerbot.policy.base import PolicyConfig
from pokerbot.ranges import RangeTracker
from pokerbot.solver.river import solve_river
from pokerbot.solver.terminal import RiverBoard

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs/data/model_error.json"
BOARDS = {
    "dry": "7h8h2cKd3s",
    "three hearts": "AhKhQh5c2d",
    "four to a straight": "9s8s7d6c2h",
    "paired": "KsKd7c4h2s",
}
HANDS = {
    "dry": ["KcQc", "9c9d", "AsQd", "6s5s"],
    "three hearts": ["JhTh", "AsKs", "QcJc", "7c6c"],
    "four to a straight": ["JcTc", "AsAd", "KhQh", "3c3d"],
    "paired": ["KhQh", "AcAd", "7h6h", "QcJc"],
}
LINES = [
    ("first to act", dict(pot=600, position="bb")),
    ("checked to", dict(pot=600, position="button")),
    ("facing 2/3 pot", dict(pot=1000, to_call=400, position="button")),
    ("facing a pot-sized bet", dict(pot=1800, to_call=600, position="bb")),
]
RICH = BetSizing(bet_fracs=(0.33, 0.75, 1.5), raise_fracs=(0.5, 1.0), max_raises=3)
SOLVE = dict(time_budget=60.0, max_iterations=600, target_pct_pot=0.1)


def grouped(solution, combo) -> np.ndarray:
    """A hand's strategy as (fold, check or call, bet or raise)."""
    out = np.zeros(3)
    for action, p in zip(solution.actions, solution.strategy[combo]):
        out[0 if action.kind == FOLD else 1 if action.kind in (CHECK, CALL) else 2] += p
    return out


def range_distance(base, variant, weights) -> float:
    """Average total-variation distance over the hero's range."""
    total = 0.0
    for combo in np.flatnonzero(weights > 0):
        total += weights[combo] * 0.5 * np.abs(grouped(base, combo) - grouped(variant, combo)).sum()
    return float(total / weights.sum())


def main():
    rows = []
    for board_name, board_text in BOARDS.items():
        board = parse_cards(board_text)
        live = RiverBoard(board).live
        for hand in HANDS[board_name]:
            for line, kw in LINES:
                spot = spot_from_partial(parse_cards(hand), board, **kw)
                combo = combo_index(*spot.hole)
                trackers = {
                    "base": RangeTracker(PolicyConfig()),
                    "soft": RangeTracker(PolicyConfig(version="soft", epsilon=0.3)),
                }
                for tracker in trackers.values():
                    tracker.catch_up(spot)
                base_ranges = tuple(trackers["base"].ranges)
                uniform = list(base_ranges)
                uniform[1 - spot.seat] = live.astype(np.float64)
                cases = {
                    "base": (base_ranges, None),
                    "soft": (tuple(trackers["soft"].ranges), None),
                    "uniform": (tuple(uniform), None),
                    "sizes": (base_ranges, RICH),
                }
                solved = {
                    name: solve_river(spot, ranges, sizing=sizing, **SOLVE)
                    for name, (ranges, sizing) in cases.items()
                }
                hero_weights = np.where(live, base_ranges[spot.seat], 0.0)
                row = {"board": board_name, "hand": hand, "line": line}
                for name, solution in solved.items():
                    mix = grouped(solution, combo)
                    row[name] = {
                        "fold_call_raise": [round(float(x), 3) for x in mix],
                        "top": ["fold", "check/call", "bet/raise"][int(np.argmax(mix))],
                        "exploitability_pct_pot": round(solution.exploitability_pct_pot, 3),
                        "hand_distance": round(
                            float(0.5 * np.abs(mix - grouped(solved["base"], combo)).sum()), 3
                        ),
                        "range_distance": round(
                            range_distance(solved["base"], solution, hero_weights), 3
                        ),
                    }
                rows.append(row)
                print(
                    f"{board_name:18s} {hand} {line:22s} "
                    + "  ".join(f"{n}:{row[n]['top']}" for n in cases)
                )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))

    n = len(rows)
    print(f"\n{n} situations")
    print(
        "| assumption changed | top action changes | mean distance, hero's hand | mean distance, hero's range |"
    )
    print("|---|---|---|---|")
    for name in ("soft", "uniform", "sizes"):
        flips = sum(r[name]["top"] != r["base"]["top"] for r in rows)
        hand = np.mean([r[name]["hand_distance"] for r in rows])
        whole = np.mean([r[name]["range_distance"] for r in rows])
        print(f"| {name} | {flips} of {n} ({flips / n:.0%}) | {hand:.3f} | {whole:.3f} |")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
