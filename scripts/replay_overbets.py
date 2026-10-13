"""Replay the biggest bets from a Slumbot log under other policy settings.

For every logged spot where the bot bet 100 big blinds or more, rebuild the
situation exactly as it was and ask what each variant would do there. This
compares variants on real situations without the noise of playing new hands.

    uv run python scripts/replay_overbets.py docs/data/slumbot_solver_no_floor.jsonl
"""

import json
import pathlib
import re
import sys
from dataclasses import replace

from pokerbot.advisor import SolverConfig, advise
from pokerbot.arena.slumbot import CONFIG, spot_from_response
from pokerbot.game.state import RAISE
from pokerbot.game.tree import BetSizing
from pokerbot.policy.base import PolicyConfig

BUDGET = dict(time_budget=30.0, max_iterations=300, target_pct_pot=0.25)
OLD_TREE = BetSizing(bet_fracs=(0.33, 0.75), all_in_max_pot=None)
# name: (range floor, bet tree)
VARIANTS = {
    "no guards": (0.0, OLD_TREE),
    "range floor 0.1": (0.1, OLD_TREE),
    "range floor 0.2": (0.2, OLD_TREE),
    "floor 0.2 and all-in limit": (0.2, BetSizing()),
}
_TOKEN = re.compile(r"b\d+|[kcf]|/")


def main():
    path = pathlib.Path(sys.argv[1])
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    bb = CONFIG.big_blind
    rows = []
    for i, record in enumerate(records):
        tokens = _TOKEN.findall(record["action"])
        for cut, token in enumerate(tokens):
            if not token.startswith("b"):
                continue
            prefix = "".join(tokens[:cut])
            n_board = [0, 3, 4, 5][prefix.count("/")]
            data = dict(record, action=prefix, board=record["board"][:n_board])
            spot = spot_from_response(data)
            if spot is None:  # the opponent's bet, not ours
                continue
            added = int(token[1:]) - spot.street_contrib[spot.seat]
            if added < 100 * bb:
                continue
            row = {
                "hand": i,
                "cards": "".join(record["hole_cards"]),
                "board": "".join(data["board"]),
            }
            row["pot_bb"] = spot.pot / bb
            row["bet_bb"] = added / bb
            row["result_bb"] = record["winnings"] / bb
            all_in = spot.legal().max_raise_to
            for name, (floor, sizing) in VARIANTS.items():
                cfg = replace(PolicyConfig(), version=name, range_floor=floor)
                rec = advise(spot, cfg, solver=SolverConfig(sizing=sizing, **BUDGET))
                row[name] = sum(
                    p
                    for action, p in rec.probabilities.items()
                    if action.startswith(RAISE) and action.endswith(str(all_in))
                )
            rows.append(row)

    names = list(VARIANTS)
    print("All-in share of the hand's strategy, by variant\n")
    print(
        "| Hand | Cards | Board | Pot, bb | Bet made, bb | Result, bb | " + " | ".join(names) + " |"
    )
    print("|---|---|---|---|---|---|" + "---|" * len(names))
    for r in rows:
        shares = " | ".join(f"{r[name]:.0%}" for name in names)
        print(
            f"| {r['hand']} | {r['cards']} | {r['board']} | {r['pot_bb']:.0f} | "
            f"{r['bet_bb']:.0f} | {r['result_bb']:+.0f} | {shares} |"
        )
    means = " | ".join(f"{sum(r[n] for r in rows) / max(len(rows), 1):.0%}" for n in names)
    print(f"| | | | | | average | {means} |")
    out = path.with_name(path.stem + "_overbet_replay.json")
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
