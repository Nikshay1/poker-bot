"""Build the 169 x 169 preflop all-in equity table by Monte Carlo.

Entry [i, j] is the equity of hand class i against hand class j when both are
all-in before the flop, averaged over the suit combinations that do not share
a card. The result is deterministic for a given seed and sample count.

    uv run python scripts/build_preflop_equity.py
"""

import pathlib
import time

import numpy as np
from numba import njit, prange

from pokerbot.cards import COMBO_CARDS, COMBO_CLASS, N_CLASSES
from pokerbot.evaluator import evaluate

SAMPLES = 200_000
SEED = 2026
OUT = pathlib.Path(__file__).resolve().parents[1] / "src/pokerbot/data/preflop_equity.npy"


@njit(cache=True, parallel=True)
def build(class_combos, class_count, combo_cards, samples, seed):
    n = class_combos.shape[0]
    table = np.full((n, n), 0.5)
    n_pairs = n * (n - 1) // 2
    for pair in prange(n_pairs):
        # unrank the pair index into i < j
        j = int((1.0 + np.sqrt(1.0 + 8.0 * pair)) / 2.0)
        while j * (j - 1) // 2 > pair:
            j -= 1
        while (j + 1) * j // 2 <= pair:
            j += 1
        i = pair - j * (j - 1) // 2
        np.random.seed(seed + pair)
        a = np.empty(7, dtype=np.int64)
        b = np.empty(7, dtype=np.int64)
        deck = np.empty(52, dtype=np.int64)
        total = 0.0
        done = 0
        while done < samples:
            ka = class_combos[i, np.random.randint(class_count[i])]
            kb = class_combos[j, np.random.randint(class_count[j])]
            a0, a1 = combo_cards[ka, 0], combo_cards[ka, 1]
            b0, b1 = combo_cards[kb, 0], combo_cards[kb, 1]
            if a0 == b0 or a0 == b1 or a1 == b0 or a1 == b1:
                continue
            m = 0
            for c in range(52):
                if c != a0 and c != a1 and c != b0 and c != b1:
                    deck[m] = c
                    m += 1
            for t in range(5):
                r = t + np.random.randint(m - t)
                deck[t], deck[r] = deck[r], deck[t]
                a[t] = deck[t]
                b[t] = deck[t]
            a[5], a[6] = a0, a1
            b[5], b[6] = b0, b1
            sa = evaluate(a)
            sb = evaluate(b)
            if sa > sb:
                total += 1.0
            elif sa == sb:
                total += 0.5
            done += 1
        table[i, j] = total / samples
        table[j, i] = 1.0 - total / samples
    return table


def main():
    class_count = np.bincount(COMBO_CLASS, minlength=N_CLASSES).astype(np.int64)
    class_combos = np.zeros((N_CLASSES, 12), dtype=np.int64)
    fill = np.zeros(N_CLASSES, dtype=np.int64)
    for k, c in enumerate(COMBO_CLASS):
        class_combos[c, fill[c]] = k
        fill[c] += 1
    start = time.perf_counter()
    table = build(class_combos, class_count, COMBO_CARDS, SAMPLES, SEED)
    print(f"built in {time.perf_counter() - start:.1f} s, {SAMPLES} samples per pair")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, table.astype(np.float32))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
