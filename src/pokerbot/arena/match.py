"""Duplicate matches between two bots, with confidence intervals.

A duplicate pair plays the same deck twice with the seats swapped, so the
luck of the cards largely cancels. The unit of analysis is the pair: the
two hands of a pair are correlated, so the interval is computed over pair
results, never over single hands.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context

import numpy as np

from pokerbot.bots.basic import Bot, make_bot
from pokerbot.game.state import GameConfig, HandState


def play_hand(bot0: Bot, bot1: Bot, config: GameConfig, deck) -> tuple[tuple[int, int], dict]:
    """Play one hand with ``bot0`` on the button. Returns payoffs and counters."""
    state = HandState(config, tuple(deck))
    bots = (bot0, bot1)
    stats = {"illegal": [0, 0], "decisions": [0, 0], "seconds": [0.0, 0.0]}
    for bot in bots:
        bot.new_hand()
    while not state.finished:
        seat = state.to_act
        legal = state.legal()
        started = time.perf_counter()
        action = bots[seat].act(state.view(seat))
        stats["seconds"][seat] += time.perf_counter() - started
        stats["decisions"][seat] += 1
        if not legal.allows(action):
            # Count it and substitute the passive action, so one bad bot
            # cannot stop a match. A non-zero count is a bug to fix.
            stats["illegal"][seat] += 1
            action = legal.passive()
        state.apply(action)
    return state.payoffs, stats


@dataclass
class MatchResult:
    bot_a: str
    bot_b: str
    pairs: int
    hands: int
    bb_per_100: float  # bot_a's win rate
    ci95: float  # half-width of the 95% interval, in bb/100
    stdev_pair_bb: float  # standard deviation of a pair's mean result, in bb per hand
    stdev_hand_bb: float  # standard deviation of a single hand, in bb
    illegal_a: int
    illegal_b: int
    ms_per_decision_a: float
    ms_per_decision_b: float
    seconds: float
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        low, high = self.bb_per_100 - self.ci95, self.bb_per_100 + self.ci95
        return (
            f"{self.bot_a} vs {self.bot_b}: {self.bb_per_100:+.1f} bb/100 "
            f"(95% CI {low:+.1f} to {high:+.1f}) over {self.hands} hands in {self.pairs} "
            f"duplicate pairs; {self.ms_per_decision_a:.1f} / {self.ms_per_decision_b:.1f} ms "
            f"per decision; illegal actions {self.illegal_a} / {self.illegal_b}; "
            f"{self.seconds:.0f} s"
        )


def _play_pairs(args):
    name_a, name_b, config, seed, first, count = args
    out = []
    for pair in range(first, first + count):
        rng = np.random.default_rng([seed, pair])
        deck = rng.permutation(52)[:9]
        # Fresh bots per pair: nothing learned on the first hand can leak
        # into the replay of the same cards.
        seeds = rng.integers(0, 2**31, size=4)
        a0, b0 = make_bot(name_a, int(seeds[0])), make_bot(name_b, int(seeds[1]))
        payoffs1, stats1 = play_hand(a0, b0, config, deck)
        a1, b1 = make_bot(name_a, int(seeds[2])), make_bot(name_b, int(seeds[3]))
        payoffs2, stats2 = play_hand(b1, a1, config, deck)
        out.append(
            (
                payoffs1[0],
                payoffs2[1],
                stats1["illegal"][0] + stats2["illegal"][1],
                stats1["illegal"][1] + stats2["illegal"][0],
                stats1["decisions"][0] + stats2["decisions"][1],
                stats1["decisions"][1] + stats2["decisions"][0],
                stats1["seconds"][0] + stats2["seconds"][1],
                stats1["seconds"][1] + stats2["seconds"][0],
            )
        )
    return out


_SINGLE_THREAD = ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _worker_init():
    # Each worker is one process; nested Numba threads would oversubscribe the CPU.
    import numba

    numba.set_num_threads(1)


def run_match(
    bot_a: str,
    bot_b: str,
    pairs: int,
    seed: int = 0,
    config: GameConfig | None = None,
    workers: int = 1,
) -> MatchResult:
    """Play ``pairs`` duplicate pairs and report bot_a's win rate."""
    config = config or GameConfig()
    started = time.perf_counter()
    if workers <= 1:
        rows = _play_pairs((bot_a, bot_b, config, seed, 0, pairs))
    else:
        chunk = max(1, math.ceil(pairs / (workers * 4)))
        jobs = [
            (bot_a, bot_b, config, seed, first, min(chunk, pairs - first))
            for first in range(0, pairs, chunk)
        ]
        rows = []
        # One thread per worker process. Without this each worker's math
        # library starts a thread per core and the workers fight each other.
        saved = {name: os.environ.get(name) for name in _SINGLE_THREAD}
        os.environ.update(dict.fromkeys(_SINGLE_THREAD, "1"))
        try:
            with ProcessPoolExecutor(workers, get_context("spawn"), _worker_init) as pool:
                for part in pool.map(_play_pairs, jobs):
                    rows.extend(part)
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    data = np.array(rows, dtype=np.float64)
    bb = config.big_blind
    hand_bb = data[:, :2] / bb
    pair_mean = hand_bb.mean(axis=1)
    n = len(pair_mean)
    stdev_pair = float(pair_mean.std(ddof=1)) if n > 1 else float("nan")
    stderr = stdev_pair / math.sqrt(n) if n > 1 else float("nan")
    decisions_a, decisions_b = data[:, 4].sum(), data[:, 5].sum()
    return MatchResult(
        bot_a=bot_a,
        bot_b=bot_b,
        pairs=n,
        hands=2 * n,
        bb_per_100=float(pair_mean.mean() * 100),
        ci95=float(1.96 * stderr * 100),
        stdev_pair_bb=stdev_pair,
        stdev_hand_bb=float(hand_bb.std(ddof=1)) if n > 1 else float("nan"),
        illegal_a=int(data[:, 2].sum()),
        illegal_b=int(data[:, 3].sum()),
        ms_per_decision_a=float(1000 * data[:, 6].sum() / max(decisions_a, 1)),
        ms_per_decision_b=float(1000 * data[:, 7].sum() / max(decisions_b, 1)),
        seconds=time.perf_counter() - started,
        seed=seed,
    )
