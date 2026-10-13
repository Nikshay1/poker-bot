"""Summarise a Slumbot session log: result, and how the bot bet.

The win rate from a few hundred hands is noise. The betting profile is the
useful part: it shows how often the bot made very large bets, which is where
an overconfident read of the opponent's range costs the most.

    uv run python scripts/slumbot_report.py runs/slumbot_solver.jsonl
"""

import json
import pathlib
import sys

from pokerbot.arena.slumbot import CONFIG, parse_actions, summarise
from pokerbot.cards import card_from_str
from pokerbot.game.state import RAISE, STREET_NAMES, Action, HandState


def hero_bets(record: dict) -> list[tuple[int, float, int]]:
    """Every bet or raise the bot made: (street, size as a multiple of the pot, chips added)."""
    seat = 1 - int(record["client_pos"])
    hole = tuple(card_from_str(c) for c in record["hole_cards"])
    other = tuple(card_from_str(c) for c in record.get("bot_hole_cards") or [])
    board = tuple(card_from_str(c) for c in record["board"])
    known = set(hole) | set(other) | set(board)
    filler = [c for c in range(52) if c not in known]
    if len(other) != 2:
        other, filler = tuple(filler[:2]), filler[2:]
    holes = (hole, other) if seat == 0 else (other, hole)
    deck = holes[0] + holes[1] + board + tuple(filler[: 5 - len(board)])
    state = HandState(CONFIG, deck)
    bets = []
    for kind, amount in parse_actions(record["action"]):
        if state.finished:
            break
        if kind == RAISE and state.to_act == seat:
            added = amount - state.street_contrib[seat]
            bets.append((state.street, added / state.pot, added))
        state.apply(Action(kind, amount))
    return bets


def main():
    path = pathlib.Path(sys.argv[1])
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = summarise([r["winnings"] for r in records])
    low, high = result["bb_per_100"] - result["ci95"], result["bb_per_100"] + result["ci95"]
    print(f"Hands: {result['hands']}")
    print(f"Win rate: {result['bb_per_100']:+.1f} bb/100 (95% CI {low:+.1f} to {high:+.1f})")
    print(f"Standard deviation: {result['stdev_hand_bb']:.1f} bb per hand")

    bb = CONFIG.big_blind
    all_bets = [(i, *bet) for i, r in enumerate(records) for bet in hero_bets(r)]
    overbets = [b for b in all_bets if b[1] > 0 and b[2] >= 2.0]
    huge = [b for b in all_bets if b[3] >= 100 * bb]
    stacked = sum(1 for r in records if abs(r["winnings"]) >= 150 * bb)
    print(f"Bets and raises by the bot: {len(all_bets)}")
    print(f"  after the flop, more than twice the pot: {len(overbets)}")
    print(f"  putting in 100 big blinds or more at once: {len(huge)}")
    print(f"Hands won or lost for 150 big blinds or more: {stacked}")
    for i, street, size, added in huge:
        r = records[i]
        print(
            f"  hand {i}: {STREET_NAMES[street]} bet of {added / bb:.0f} bb ({size:.1f}x pot) with "
            f"{''.join(r['hole_cards'])} on {''.join(r['board'])}, result {r['winnings'] / bb:+.0f} bb"
        )


if __name__ == "__main__":
    main()
