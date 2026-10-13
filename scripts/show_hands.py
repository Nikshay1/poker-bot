"""Print hands played between two bots, one line per action.

Reading real hands is the quickest way to find a policy that does something
silly.

    uv run python scripts/show_hands.py heuristic equity --hands 10 --seed 3
"""

import argparse

import numpy as np

from pokerbot.bots.basic import make_bot
from pokerbot.cards import cards_to_str
from pokerbot.game.state import STREET_NAMES, GameConfig, HandState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("button")
    parser.add_argument("big_blind")
    parser.add_argument("--hands", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stack-bb", type=int, default=100)
    args = parser.parse_args()

    config = GameConfig(stacks=(args.stack_bb * 100, args.stack_bb * 100))
    names = (args.button, args.big_blind)
    total = 0
    for hand in range(args.hands):
        rng = np.random.default_rng([args.seed, hand])
        deck = rng.permutation(52)[:9]
        bots = (
            make_bot(names[0], int(rng.integers(2**31))),
            make_bot(names[1], int(rng.integers(2**31))),
        )
        state = HandState(config, tuple(deck))
        for bot in bots:
            bot.new_hand()
        print(
            f"\nHand {hand}: {names[0]} (button) {cards_to_str(state.hole(0))}  "
            f"{names[1]} (big blind) {cards_to_str(state.hole(1))}"
        )
        street = -1
        while not state.finished:
            if state.street != street:
                street = state.street
                board = f" {cards_to_str(state.board)}" if state.board else ""
                print(f"  {STREET_NAMES[street]}{board}  pot {state.pot}")
            seat = state.to_act
            action = bots[seat].act(state.view())
            print(f"    {names[seat]:10s} {action}")
            state.apply(action)
        ending = (
            "folded" if state.folded is not None else f"showdown on {cards_to_str(state.board)}"
        )
        total += state.payoffs[0]
        print(f"  {ending}: {names[0]} {state.payoffs[0]:+d}, {names[1]} {state.payoffs[1]:+d}")
    print(f"\n{names[0]} total over {args.hands} hands: {total:+d} chips")


if __name__ == "__main__":
    main()
