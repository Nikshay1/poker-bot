"""Speed and latency on this machine.

The numbers set how many hands an evaluation can afford, so they are
measured, not assumed.
"""

from __future__ import annotations

import resource
import time

import numpy as np
from numba import njit

from pokerbot.cards import parse_cards
from pokerbot.evaluator import evaluate


@njit(cache=True)
def _evaluate_many(hands):
    total = 0
    for i in range(hands.shape[0]):
        total += evaluate(hands[i])
    return total


def _percentiles(values) -> dict:
    arr = np.array(values) * 1000.0
    return {
        "n": int(len(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
    }


def run_bench(quick: bool = False) -> dict:
    from pokerbot.advisor import advise, spot_from_partial
    from pokerbot.bots.basic import make_bot
    from pokerbot.equity import range_equity, uniform_range
    from pokerbot.game.state import GameConfig, HandState

    report: dict = {}

    # Cold start: the first decision in a fresh process, including loading
    # compiled code from the on-disk cache (or compiling it the first time).
    started = time.perf_counter()
    spot = spot_from_partial(parse_cards("AsKd"), parse_cards("7h8h2c"), pot=1000, to_call=400)
    advise(spot)
    report["cold_first_decision_ms"] = (time.perf_counter() - started) * 1000

    rng = np.random.default_rng(0)
    n = 200_000 if quick else 2_000_000
    hands = np.array([rng.permutation(52)[:7] for _ in range(20_000)], dtype=np.int64)
    hands = np.tile(hands, (n // len(hands), 1))
    _evaluate_many(hands[:10])
    started = time.perf_counter()
    _evaluate_many(hands)
    report["evaluator_hands_per_second"] = len(hands) / (time.perf_counter() - started)

    weights = uniform_range()
    report["range_equity_ms"] = {}
    for name, text in (("flop", "7h8h2c"), ("turn", "7h8h2cKd"), ("river", "7h8h2cKd3s")):
        board = parse_cards(text)
        range_equity(board, weights)
        reps = 5 if quick else 20
        started = time.perf_counter()
        for _ in range(reps):
            range_equity(board, weights)
        report["range_equity_ms"][name] = (time.perf_counter() - started) / reps * 1000

    # Decision latency by street, from self-play of the heuristic bot.
    config = GameConfig()
    n_hands = 100 if quick else 500
    by_street = {0: [], 1: [], 2: [], 3: []}
    started_all = time.perf_counter()
    for hand in range(n_hands):
        deck = np.random.default_rng([1, hand]).permutation(52)[:9]
        state = HandState(config, tuple(deck))
        bots = (make_bot("heuristic", 2 * hand), make_bot("heuristic", 2 * hand + 1))
        while not state.finished:
            view = state.view()
            started = time.perf_counter()
            action = bots[state.to_act].act(view)
            by_street[view.street].append(time.perf_counter() - started)
            state.apply(action)
    elapsed = time.perf_counter() - started_all
    names = ["preflop", "flop", "turn", "river"]
    report["heuristic_decision_latency"] = {
        names[s]: _percentiles(v) for s, v in by_street.items() if v
    }
    report["heuristic_selfplay_hands_per_second"] = n_hands / elapsed
    report["peak_memory_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    lines = [
        f"Cold first decision: {report['cold_first_decision_ms']:.0f} ms",
        f"Evaluator: {report['evaluator_hands_per_second'] / 1e6:.1f} million 7-card hands per second, one thread",
        "Range equity (all 1326 combos against a full range): "
        + ", ".join(f"{k} {v:.1f} ms" for k, v in report["range_equity_ms"].items()),
        "Heuristic decision latency (ms):",
    ]
    for name, p in report["heuristic_decision_latency"].items():
        lines.append(
            f"  {name:8s} p50 {p['p50_ms']:.1f}  p95 {p['p95_ms']:.1f}  "
            f"p99 {p['p99_ms']:.1f}  max {p['max_ms']:.1f}  (n={p['n']})"
        )
    lines.append(
        f"Heuristic self-play: {report['heuristic_selfplay_hands_per_second']:.1f} hands per second, one process"
    )
    lines.append(f"Peak memory: {report['peak_memory_mb']:.0f} MB")
    report["text"] = "\n".join(lines)
    return report
