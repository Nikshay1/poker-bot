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
from pokerbot.policy.base import PolicyConfig

FLOORS = (0.0, 0.1, 0.2)
SOLVER = SolverConfig(time_budget=30.0, max_iterations=300, target_pct_pot=0.25)
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
            for floor in FLOORS:
                cfg = replace(PolicyConfig(), version=f"floor{floor}", range_floor=floor)
                rec = advise(spot, cfg, solver=SOLVER)
                all_in = spot.legal().max_raise_to
                big = sum(
                    p
                    for name, p in rec.probabilities.items()
                    if name.startswith(RAISE) and name.endswith(str(all_in))
                )
                row[f"all_in_{floor}"] = big
                row[f"top_{floor}"] = max(rec.probabilities, key=rec.probabilities.get)
            rows.append(row)

    print(
        "| Hand | Cards | Board | Pot, bb | Bet made, bb | Result, bb | "
        + " | ".join(f"All-in share, floor {f}" for f in FLOORS)
        + " |"
    )
    print("|---|---|---|---|---|---|" + "---|" * len(FLOORS))
    for r in rows:
        shares = " | ".join(f"{r[f'all_in_{f}']:.0%}" for f in FLOORS)
        print(
            f"| {r['hand']} | {r['cards']} | {r['board']} | {r['pot_bb']:.0f} | "
            f"{r['bet_bb']:.0f} | {r['result_bb']:+.0f} | {shares} |"
        )
    for f in FLOORS:
        mean = sum(r[f"all_in_{f}"] for r in rows) / max(len(rows), 1)
        print(f"floor {f}: average all-in share over these {len(rows)} spots: {mean:.0%}")
    out = path.with_name(path.stem + "_overbet_replay.json")
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
