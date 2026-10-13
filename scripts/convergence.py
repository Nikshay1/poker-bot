"""Numerical convergence of the river solver.

For a few river situations, run plain CFR, CFR+ and Discounted CFR and
record exploitability (as a percentage of the pot) against iterations and
wall time. Exploitability here is exact within the bet tree and the ranges
at the root.

    uv run python scripts/convergence.py
"""

import json
import pathlib
import time

import numpy as np

from pokerbot.advisor import spot_from_partial
from pokerbot.cards import parse_cards
from pokerbot.game.tree import BetSizing, BetState
from pokerbot.ranges import RangeTracker
from pokerbot.solver.river import build_river_tree
from pokerbot.solver.terminal import RiverBoard
from pokerbot.solver.vector_cfr import VARIANTS, VectorCFR

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs/data/convergence.json"
CHECKPOINTS = [10, 25, 50, 100, 200, 400, 800]
SPOTS = [
    ("first to act, pot 600", dict(hole="AsQd", board="7h8h2cKd3s", pot=600, position="bb")),
    ("facing a 40% pot bet", dict(hole="AsQd", board="7h8h2cKd3s", pot=1000, to_call=400)),
    ("checked to, pot 2000", dict(hole="JcTc", board="AhKhQh5c2d", pot=2000)),
    ("first to act, paired board", dict(hole="AcJc", board="KsKd7c4h2s", pot=1200, position="bb")),
]


def main():
    rows = []
    for label, kw in SPOTS:
        kw = dict(kw)
        spot = spot_from_partial(parse_cards(kw.pop("hole")), parse_cards(kw.pop("board")), **kw)
        tracker = RangeTracker()
        tracker.catch_up(spot)
        board = RiverBoard(spot.board)
        reach = []
        for r in tracker.beliefs():
            r = np.where(board.live, r, 0.0)
            reach.append(r / r.sum())
        norm = float(reach[0] @ board.compatible(reach[1]))
        for variant in VARIANTS:
            tree = build_river_tree(BetState.from_spot(spot), board, BetSizing())
            solver = VectorCFR(tree, variant, (reach[0], reach[1]), norm)
            spent = 0.0
            for target in CHECKPOINTS:
                started = time.perf_counter()
                solver.iterate(target - solver.t)
                spent += time.perf_counter() - started
                rows.append(
                    {
                        "spot": label,
                        "decisions": len(tree.decisions),
                        "terminals": tree.n_terminals,
                        "variant": variant,
                        "iterations": target,
                        "seconds": round(spent, 4),
                        "exploitability_pct_pot": 100.0 * solver.exploitability() / spot.pot,
                    }
                )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))

    for label, _ in SPOTS:
        mine = [r for r in rows if r["spot"] == label]
        print(f"\n{label} ({mine[0]['decisions']} decisions, {mine[0]['terminals']} endings)")
        print("| iterations | " + " | ".join(f"{v} % pot" for v in VARIANTS) + " | dcfr seconds |")
        print("|---|" + "---|" * (len(VARIANTS) + 1))
        for target in CHECKPOINTS:
            cells = []
            for variant in VARIANTS:
                row = next(r for r in mine if r["variant"] == variant and r["iterations"] == target)
                cells.append(f"{row['exploitability_pct_pot']:.3f}")
            seconds = next(
                r["seconds"] for r in mine if r["variant"] == "dcfr" and r["iterations"] == target
            )
            print(f"| {target} | " + " | ".join(cells) + f" | {seconds:.2f} |")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
