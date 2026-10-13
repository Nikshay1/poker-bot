"""Command line: ``pokerbot advise``, ``match``, ``bench`` and ``slumbot``."""

from __future__ import annotations

import argparse
import json
import sys

from pokerbot.cards import parse_cards
from pokerbot.game.state import GameConfig, SpotError


def _advise(args) -> int:
    from pokerbot.advisor import SolverConfig, advise, spot_from_partial
    from pokerbot.policy.multiway import advise_multiway

    try:
        hole = parse_cards(args.hole)
        board = parse_cards(args.board or "")
        if len(hole) != 2:
            raise SpotError("bad_hole", "give exactly two hole cards, for example AsKd")
        if args.players > 2:
            rec = advise_multiway(
                hole,
                board,
                pot=args.pot,
                to_call=args.to_call,
                stack=args.stack,
                n_opponents=args.players - 1,
                big_blind=args.bb,
            )
            text = rec.render_multiway(hole, board, args.pot, args.to_call)
        else:
            spot = spot_from_partial(
                hole,
                board,
                pot=args.pot,
                to_call=args.to_call,
                stack=args.stack,
                position=args.position,
                small_blind=args.bb // 2,
                big_blind=args.bb,
                starting_stack_bb=args.stack_bb,
            )
            solver = None if args.no_solver else SolverConfig(time_budget=args.time)
            rec = advise(spot, solver=solver)
            text = rec.render(spot)
    except SpotError as err:
        payload = err.as_dict()
        print(json.dumps(payload) if args.json else f"error ({err.code}): {err}", file=sys.stderr)
        return 2
    except ValueError as err:
        payload = {"error": "bad_input", "message": str(err)}
        print(json.dumps(payload) if args.json else f"error: {err}", file=sys.stderr)
        return 2
    print(json.dumps(rec.to_dict(), indent=2) if args.json else text)
    return 0


def _match(args) -> int:
    from pokerbot.arena.match import run_match

    config = GameConfig(args.bb // 2, args.bb, (args.stack_bb * args.bb, args.stack_bb * args.bb))
    result = run_match(args.bot_a, args.bot_b, args.pairs, args.seed, config, args.workers)
    print(json.dumps(result.to_dict(), indent=2) if args.json else result.summary())
    return 0


def _bench(args) -> int:
    from pokerbot.bench import run_bench

    report = run_bench(quick=args.quick)
    print(json.dumps(report, indent=2) if args.json else report["text"])
    return 0


def _slumbot(args) -> int:
    from pokerbot.arena.slumbot import run_session, summarise_log

    result = run_session(args.bot, args.hands, args.seed, args.log)
    if args.log:
        result["log_total"] = summarise_log(args.log)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    def line(label, r):
        low, high = r["bb_per_100"] - r["ci95"], r["bb_per_100"] + r["ci95"]
        return (
            f"{label}: {r['bb_per_100']:+.1f} bb/100 (95% CI {low:+.1f} to {high:+.1f}) "
            f"over {r['hands']} hands"
        )

    print(line(f"{result['bot']} vs Slumbot, this session", result))
    print(f"hands dropped after a network failure: {result['dropped_hands']}")
    if args.log:
        print(line("whole log", result["log_total"]))
    print("These are single hands, not duplicate pairs.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="pokerbot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    advise = sub.add_parser("advise", help="recommend an action for one situation")
    advise.add_argument("--hole", required=True, help="your two cards, for example AsKd")
    advise.add_argument("--board", default="", help="community cards, for example 7h8h2c")
    advise.add_argument("--pot", type=int, help="chips in the pot, including any bet you face")
    advise.add_argument("--to-call", type=int, default=0, help="chips needed to call")
    advise.add_argument("--stack", type=int, help="chips you have left")
    advise.add_argument("--position", choices=["button", "bb"], default="button")
    advise.add_argument("--players", type=int, default=2, help="players in the hand")
    advise.add_argument("--bb", type=int, default=100, help="big blind in chips")
    advise.add_argument(
        "--stack-bb", type=float, default=100.0, help="starting stacks in big blinds"
    )
    advise.add_argument("--time", type=float, default=2.0, help="seconds allowed for a river solve")
    advise.add_argument(
        "--no-solver", action="store_true", help="use the heuristic on the river too"
    )
    advise.add_argument("--json", action="store_true")
    advise.set_defaults(run=_advise)

    match = sub.add_parser("match", help="play a duplicate match between two bots")
    match.add_argument("bot_a")
    match.add_argument("bot_b")
    match.add_argument("--pairs", type=int, default=1000, help="duplicate pairs (two hands each)")
    match.add_argument("--seed", type=int, default=0)
    match.add_argument("--workers", type=int, default=1)
    match.add_argument("--bb", type=int, default=100)
    match.add_argument("--stack-bb", type=int, default=100)
    match.add_argument("--json", action="store_true")
    match.set_defaults(run=_match)

    bench = sub.add_parser("bench", help="measure speed and latency on this machine")
    bench.add_argument("--quick", action="store_true")
    bench.add_argument("--json", action="store_true")
    bench.set_defaults(run=_bench)

    slumbot = sub.add_parser("slumbot", help="play against Slumbot over its public API")
    slumbot.add_argument("--bot", default="solver")
    slumbot.add_argument("--hands", type=int, default=100)
    slumbot.add_argument("--seed", type=int, default=0)
    slumbot.add_argument("--log", help="append one JSON line per hand to this file")
    slumbot.add_argument("--json", action="store_true")
    slumbot.set_defaults(run=_slumbot)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
