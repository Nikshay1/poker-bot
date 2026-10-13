"""Cross-check the river solver against an independent implementation.

The reference is Noam Brown's public river solver
(https://github.com/noambrown/poker_solver, MIT licence). It is not a
dependency; clone it somewhere and pass the path:

    git clone https://github.com/noambrown/poker_solver /tmp/poker_solver
    uv run python -I scripts/cross_check.py --ref /tmp/poker_solver

Three checks, on the same board, pot, stacks, ranges and bet sizes:

1. The two bet trees are identical, node by node and action by action.
2. This project's solution is exported into the reference code, and the
   reference's own best-response routine measures its exploitability. That
   number must equal the one this project computes for itself.
3. Both solvers are run and their game values must agree to within the two
   solutions' exploitabilities.

The reference counts payoffs from the start of the river with the pot
already in the middle, so its values are this project's plus half the pot.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

from pokerbot.advisor import spot_from_partial
from pokerbot.cards import COMBO_CARDS, N_COMBOS, combo_index, combo_to_str, parse_cards
from pokerbot.game.state import CALL, CHECK, FOLD, RAISE
from pokerbot.game.tree import BetSizing, BetState
from pokerbot.solver.public_tree import TERMINAL
from pokerbot.solver.river import build_river_tree
from pokerbot.solver.terminal import RiverBoard
from pokerbot.solver.vector_cfr import VectorCFR

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs/data/cross_check.json"
BOARD = "KsTh7s4d2s"
POT, STACK = 1000, 9500
SIZES = (0.5, 1.0)
MAX_RAISES = 2
HANDS_PER_PLAYER = 150


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="path to a clone of noambrown/poker_solver")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--ref-iterations", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    sys.path.insert(0, str(pathlib.Path(args.ref) / "python" / "src"))
    from algorithms.vector_cfr import VectorCFRConfig, VectorCFRTrainer
    from algorithms.vector_eval import (
        action_tokens,
        best_response_value,
        build_blocked_indices,
        build_strength_summary,
        exploitability,
    )
    from games.river_holdem import RiverHoldemConfig, RiverHoldemGame

    # Random weighted ranges. The two players hold the same set of hands with
    # different weights: the reference's showdown code looks each hand's
    # strength up among the opponent's hands, so it needs a shared set.
    board = parse_cards(BOARD)
    live = np.flatnonzero(RiverBoard(board).live)
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(live, size=HANDS_PER_PLAYER, replace=False)
    ranges = []  # by this project's seat: 0 is in position, 1 acts first
    for _ in (0, 1):
        weights = np.zeros(N_COMBOS)
        weights[chosen] = rng.uniform(0.05, 1.0, size=HANDS_PER_PLAYER)
        ranges.append(weights)

    # The reference's player 0 acts first, which is this project's seat 1.
    def their_range(seat):
        combos = np.flatnonzero(ranges[seat])
        return [combo_to_str(int(k)) for k in combos], [float(ranges[seat][k]) for k in combos]

    hands0, weights0 = their_range(1)
    hands1, weights1 = their_range(0)
    game = RiverHoldemGame(
        RiverHoldemConfig(
            board=[BOARD[i : i + 2] for i in range(0, 10, 2)],
            pot=POT,
            stacks=(STACK, STACK),
            bet_sizes=SIZES,
            include_all_in=True,
            max_raises=MAX_RAISES,
            ranges=(hands0, hands1),
            range_weights=(weights0, weights1),
        )
    )

    def my_combo(hand) -> int:
        # The reference numbers cards suit-major; this project rank-major.
        a, b = (4 * (c % 13) + c // 13 for c in hand.cards)
        return combo_index(a, b)

    their_combos = [[my_combo(h) for h in game.hands[p]] for p in (0, 1)]

    # This project's side of the same game.
    hole = [int(c) for c in COMBO_CARDS[live[0]]]
    spot = spot_from_partial(hole, board, pot=POT, position="bb", stack=STACK)
    sizing = BetSizing(
        bet_fracs=SIZES,
        raise_fracs=SIZES,
        all_in=True,
        max_raises=MAX_RAISES,
        commit_frac=10.0,  # never round a size up to all-in; the reference does not
        all_in_max_pot=None,  # the reference always offers all-in
    )
    river = RiverBoard(board)
    reach = [r / r.sum() for r in ranges]
    norm = float(reach[0] @ river.compatible(reach[1]))
    root_state = BetState.from_spot(spot)
    tree = build_river_tree(root_state, river, sizing)
    solver = VectorCFR(tree, "dcfr", (reach[0], reach[1]), norm)
    started = time.perf_counter()
    solver.iterate(args.iterations)
    my_seconds = time.perf_counter() - started

    # Walk both trees together: check they match and export the strategy.
    profile = {0: {}, 1: {}}
    nodes = {"decisions": 0, "terminals": 0}

    def walk(node, state, theirs):
        if node.kind == TERMINAL:
            assert game.is_terminal(theirs), "trees differ: one ends where the other continues"
            nodes["terminals"] += 1
            return
        nodes["decisions"] += 1
        player = game.current_player(theirs)
        seat = node.player
        assert seat == 1 - player, "trees differ: a different player is to act"
        their_actions = game.legal_actions(theirs)
        assert len(their_actions) == len(node.actions), "trees differ: number of actions"
        index = []
        for action in their_actions:
            if action.label == "f":
                match = [i for i, a in enumerate(node.actions) if a.kind == FOLD]
            elif action.label == "c":
                match = [i for i, a in enumerate(node.actions) if a.kind in (CHECK, CALL)]
            else:
                # Their amount is the chips added beyond the call.
                level = state.street_contrib[1 - seat]
                match = [
                    i
                    for i, a in enumerate(node.actions)
                    if a.kind == RAISE and a.amount - level == action.amount
                ]
            assert len(match) == 1, f"trees differ: no match for {action} among {node.actions}"
            index.append(match[0])
        average = solver.average_strategy(node)
        matrix = [[float(average[k, i]) for i in index] for k in their_combos[player]]
        profile[player][game.infoset_key(theirs, player)] = (action_tokens(their_actions), matrix)
        for action, i in zip(their_actions, index):
            walk(node.children[i], state.apply(node.actions[i]), game.next_state(theirs, action))

    walk(tree.root, root_state, game.initial_state())
    assert nodes["decisions"] == len(tree.decisions) and nodes["terminals"] == tree.n_terminals

    summaries = {0: build_strength_summary(game.hands[1]), 1: build_strength_summary(game.hands[0])}
    blocked = {p: build_blocked_indices(game.hands[p], summaries[p]) for p in (0, 1)}

    # Check 2: the reference's best response against this project's strategy.
    ref_br = [
        best_response_value(game, p, profile[1 - p], summaries[p], blocked[p]) for p in (0, 1)
    ]
    ref_exploitability = exploitability(game, profile, summaries, blocked)
    my_br = [solver.best_response_value(1), solver.best_response_value(0)]  # by their player
    my_exploitability = solver.exploitability()

    # Check 3: the reference solving the game itself.
    trainer = VectorCFRTrainer(
        game,
        VectorCFRConfig(
            use_plus=False,
            linear_weighting=False,
            alternating=True,
            use_dcfr=True,
            dcfr_alpha=1.5,
            dcfr_beta=0.0,
            dcfr_gamma=2.0,
        ),
    )
    started = time.perf_counter()
    trainer.run(args.ref_iterations)
    ref_seconds = time.perf_counter() - started
    ref_profile = trainer.average_strategy_profile()
    ref_own_br = [
        best_response_value(game, p, ref_profile[1 - p], summaries[p], blocked[p]) for p in (0, 1)
    ]
    ref_own_exploitability = exploitability(game, ref_profile, summaries, blocked)
    # Each solution brackets the first player's game value between what the
    # opponent's best response leaves it and its own best response.
    my_bracket = (POT - (my_br[1] + POT / 2), my_br[0] + POT / 2)
    ref_bracket = (POT - ref_own_br[1], ref_own_br[0])

    report = {
        "board": BOARD,
        "pot": POT,
        "stack": STACK,
        "bet_sizes": SIZES,
        "max_raises": MAX_RAISES,
        "hands_per_player": HANDS_PER_PLAYER,
        "decisions": nodes["decisions"],
        "terminals": nodes["terminals"],
        "iterations": args.iterations,
        "seconds": round(my_seconds, 2),
        "exploitability_own_measure": my_exploitability,
        "exploitability_reference_measure": ref_exploitability,
        "best_response_values_own_plus_half_pot": [v + POT / 2 for v in my_br],
        "best_response_values_reference": ref_br,
        "first_player_value_bracket": my_bracket,
        "reference_iterations": args.ref_iterations,
        "reference_seconds": round(ref_seconds, 2),
        "reference_exploitability": ref_own_exploitability,
        "reference_first_player_value_bracket": ref_bracket,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))

    print(f"Trees match: {nodes['decisions']} decisions, {nodes['terminals']} endings")
    print(f"Exploitability of this project's solution after {args.iterations} iterations:")
    print(f"  measured by this project:  {my_exploitability:.9f} chips")
    print(f"  measured by the reference: {ref_exploitability:.9f} chips")
    for p in (0, 1):
        print(
            f"  best response of player {p}: own {my_br[p] + POT / 2:.6f}, reference {ref_br[p]:.6f}"
        )
    print("Value of the game to the first player, in the reference's units:")
    print(f"  this project:  between {my_bracket[0]:.4f} and {my_bracket[1]:.4f}")
    print(
        f"  the reference: between {ref_bracket[0]:.4f} and {ref_bracket[1]:.4f} "
        f"({args.ref_iterations} iterations, exploitability {ref_own_exploitability:.4f})"
    )
    print(f"Time: {my_seconds:.1f} s here, {ref_seconds:.1f} s for the reference's Python solver")
    agree = abs(my_exploitability - ref_exploitability) < 1e-6 * POT
    overlap = my_bracket[0] <= ref_bracket[1] + 1e-6 and ref_bracket[0] <= my_bracket[1] + 1e-6
    print("PASS" if agree and overlap else "FAIL")
    print(f"wrote {OUT}")
    return 0 if agree and overlap else 1


if __name__ == "__main__":
    sys.exit(main())
