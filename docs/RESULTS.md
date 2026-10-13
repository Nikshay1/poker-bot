# Results

Everything here was measured on one laptop (16 threads, 15 GB RAM). Each number can be reproduced with the command given next to it. The raw data is in [`docs/data/`](data).

## Summary

1. **The solver is correct.** It reproduces the known solutions of Kuhn and Leduc poker, matches the published push-or-fold chart, and an independent solver's best-response code reports the same exploitability for its river solutions to nine decimal places.
2. **The river solver earns chips.** Adding it to the heuristic bot wins **+39.1 bb/100** (95% interval +26.4 to +51.9) over 20,000 hands in a head-to-head duplicate match.
3. **Solver precision is not the limit; the range estimate is.** A river solve reaches about 0.1% of the pot in exploitability. But changing the assumed opponent range changes the recommended action in 20% to 30% of test situations, while a richer bet tree changes it in 3%.
4. **Both bots beat all three baselines** by wide margins, with no illegal action in 50,000 evaluation hands.
5. **Against Slumbot** the sample is too small to say anything about strength. See the section below for what it does and does not show.

Terms used below:

- **bb/100**: big blinds won per 100 hands.
- **Duplicate pair**: the same cards played twice with the seats swapped, so card luck largely cancels. Intervals are computed over pairs.
- **Exploitability**: how much a perfect opponent would gain against a strategy, averaged over both seats. Zero means an equilibrium. Here it is always measured inside a stated game: a given bet tree and given starting ranges.

## 1. Correctness

`uv run pytest` runs 118 checks in about 15 seconds; `uv run pytest -m slow` adds the exhaustive one.

| What | Check | Result |
|---|---|---|
| Hand evaluator | Category counts over all 2,598,960 five-card hands | Exact match, and exactly 7,462 distinct values |
| | Category counts over all 133,784,560 seven-card hands | Exact match |
| | Against `phevaluator` on 20,000 random pairs of hands | Same ordering every time |
| | 7-card value equals the best 5-card subset | 3,000 random hands |
| Equity | Fast range method against a direct all-pairs sum | Equal to 1e-9 on river and turn boards |
| Rules engine | Chips conserved at every step | 3,000 random hands at three stack depths |
| | Solver's bet tree against the engine | 400 random river lines agree |
| Kuhn poker | First player's value | -0.05556, the known -1/18 |
| | Equilibrium structure | Bets the King 3x as often as it bluffs the Jack (0.614 and 0.205), as theory requires |
| Leduc poker | First player's value | -0.0856, the published value |
| River solver | Fast fold and showdown values against the 1326 x 1326 matrix product | Equal to 1e-8 |
| | Polarized range against bluff-catchers, bets of 0.5, 1 and 2 pots | Bluffs `s/(1+s)` per value hand, within 0.01 |
| Push-or-fold | 10 big blinds | 58.3% all-in, 37.4% call; published charts give about 58% and 37% |

### Cross-check against an independent solver

`uv run python -I scripts/cross_check.py --ref <clone of noambrown/poker_solver>`

The reference is Noam Brown's public river solver. The script builds the same game in both (board Ks Th 7s 4d 2s, pot 1000, stacks 9500, bets and raises of 0.5 and 1 pot plus all-in, two raises, 150 hands per player with random weights) and makes three comparisons.

| Comparison | This project | Reference |
|---|---|---|
| Bet tree | 20 decisions, 37 endings | Identical, node by node |
| Exploitability of this project's solution (2,000 iterations), in chips | 0.039305939 | 0.039305939, from the reference's own best-response code |
| Value of the game to the first player | between 455.42 and 455.50 | between 454.16 and 456.51, from the reference's own solve (300 iterations) |

The second row is the strong one. The strategy was exported into the reference program, and code written by someone else measured how exploitable it is. The two measurements agree to nine decimal places.

## 2. Numerical convergence

`uv run python scripts/convergence.py`

Exploitability as a percentage of the pot, on four river situations with tracked ranges and the default bet tree. Each cell lists the four situations in order: first to act (24 decisions), facing a bet (6), checked to (10), first to act on a paired board (22).

| Iterations | Plain CFR | CFR+ | Discounted CFR | DCFR time, seconds |
|---|---|---|---|---|
| 50 | 15.8 / 1.34 / 2.07 / 4.20 | 6.40 / 0.88 / 1.21 / 2.95 | 3.33 / 2.41 / 0.71 / 1.58 | 0.10 / 0.03 / 0.04 / 0.10 |
| 100 | 8.26 / 0.65 / 1.19 / 2.22 | 2.75 / 0.53 / 0.56 / 1.00 | 1.21 / 0.53 / 0.34 / 0.39 | 0.21 / 0.05 / 0.08 / 0.19 |
| 200 | 4.05 / 0.32 / 0.70 / 1.12 | 1.01 / 0.17 / 0.26 / 0.29 | 0.53 / 0.13 / 0.11 / 0.10 | 0.41 / 0.10 / 0.17 / 0.37 |
| 400 | 2.03 / 0.21 / 0.37 / 0.55 | 0.44 / 0.06 / 0.09 / 0.08 | 0.19 / 0.03 / 0.03 / 0.03 | 0.83 / 0.21 / 0.34 / 0.73 |
| 800 | 1.06 / 0.11 / 0.19 / 0.30 | 0.19 / 0.02 / 0.03 / 0.02 | 0.06 / 0.01 / 0.01 / 0.01 | 1.65 / 0.41 / 0.67 / 1.45 |

What this shows:

- Discounted CFR is the fastest from 100 iterations on, in all four situations. At 800 iterations it is 9 to 50 times more accurate than plain CFR. In the first 50 iterations it is not always ahead.
- A solve of about half a second reaches 0.1% to 0.5% of the pot. The match bot stops at 0.25% of the pot or 300 iterations.
- Bigger trees converge more slowly per iteration and cost more per iteration.

On the small games the same pattern holds. After 1,000 iterations on Leduc poker, exploitability is 0.0118 for plain CFR, 0.00026 for CFR+ and 0.00017 for Discounted CFR, in units of the ante.

## 3. Model error

`uv run python scripts/model_error.py`

A solve can be exact and still answer the wrong question, because the opponent's range and the bet sizes are assumptions. This experiment holds 64 river situations fixed (4 boards, 4 hands, 4 betting lines) and changes one assumption at a time. Every solve reached about 0.09% of the pot, so the differences below are not numerical noise.

| Assumption changed | Top action changes | Distance, hero's hand | Distance, hero's whole range |
|---|---|---|---|
| Range inferred less sharply (uniform component 0.3 instead of 0.1) | 13 of 64 (20%) | 0.16 | 0.16 |
| Opponent may hold any hand | 19 of 64 (30%) | 0.30 | 0.38 |
| Richer bet tree (3 bet sizes, 2 raise sizes) | 2 of 64 (3%) | 0.03 | 0.05 |

Distance is total-variation distance between strategies, grouped into fold, check or call, and bet or raise: 0 means identical, 1 means completely different.

One example makes the point. With pocket aces on a board of 9s 8s 7d 6c 2h, facing a bet of two thirds of the pot:

| Opponent range | Solver's answer for the aces |
|---|---|
| The model's estimate | Raise, 100% |
| Any hand | Call, 100% |

Both answers are numerically exact for their own inputs. They differ because the model reasons that a hand strong enough to beat aces would have bet an earlier street, so it gives such hands little weight. The uniform range does not. A human player would call here, because any ten or five makes a straight.

What follows from this:

- The next improvement should go into estimating ranges, not into more solver iterations or more bet sizes.
- The river solver's 0.1% figure must not be read as the accuracy of the bot's decisions.

## 4. Match results

`uv run python scripts/evaluate.py --seed 1000` then `uv run python scripts/report.py docs/data/matches_seed1000.json`

Final evaluation: seed 1000, which was not used for anything else, with the bots frozen at commit `afbdae9`. Stacks are 100 big blinds, reset every hand.

| Bot | Opponent | Win rate, bb/100 | 95% interval | Hands |
|---|---|---|---|---|
| solver | heuristic | +39.1 | +26.4 to +51.9 | 20,000 |
| solver | equity | +226.9 | +174.3 to +279.5 | 6,000 |
| solver | call | +572.8 | +434.9 to +710.7 | 3,000 |
| solver | random | +330.6 | +254.8 to +406.3 | 3,000 |
| heuristic | equity | +160.7 | +125.5 to +195.8 | 6,000 |
| heuristic | call | +287.8 | +250.6 to +324.9 | 3,000 |
| heuristic | random | +306.8 | +237.2 to +376.4 | 3,000 |
| equity | call | +201.7 | +183.3 to +220.0 | 3,000 |
| equity | random | +582.4 | +506.3 to +658.5 | 3,000 |

Illegal actions across all 50,000 hands: 0.

The bots:

- **solver**: the heuristic bot with river decisions solved.
- **heuristic**: the heuristic advisor on every street.
- **equity**: plays its equity against a random hand, ignoring what the betting says. This is the natural first attempt at the task.
- **call**: never folds, never raises.
- **random**: picks among fold, check or call, and a random raise.

How to read the table:

- **The first row is the ablation.** The two bots differ only on the river, so +39.1 bb/100 is what the river solver adds. The interval excludes zero by a wide margin.
- The heuristic bot beats the equity bot by 161 bb/100. Reading the opponent's range from the betting is worth more than raw hand strength.
- These baselines are weak. Beating them shows the bots are sound, not that they are strong. Neither bot adapts to its opponent: the equity bot beats the random bot by more than the solver does, because a bot that simply bets its good hands happens to exploit random play well.
- Win rates against the call and random bots have wide intervals because those opponents put in huge pots at random.

### Tuning record

Tuning used seed 200 only. One change was adopted.

| Variant | Against the base heuristic | Against the equity bot |
|---|---|---|
| Base (extra bluffs on flop and turn) | | +101.2 (+53.5 to +148.9) |
| Bluff at the theoretical ratio on every street | +53.8 (+22.1 to +85.5) | +154.7 (+110.9 to +198.4) |
| Range inferred less sharply (0.3) | -48.2 (-82.1 to -14.3) | +99.5 (+51.9 to +147.0) |

- The first guess multiplied the theoretical bluff ratio by 2 on the flop and 1.5 on the turn. It was a leak, and both comparisons say so, so the multipliers were set to 1.
- Softer range inference was no better against the equity bot. Its loss against the base heuristic is not strong evidence, because in that match the opponent plays exactly the model that sharp inference assumes.

An earlier ablation on seed 100, before the tuning, gave solver against heuristic +35.0 (-2.8 to +72.7) over 4,000 hands: the same direction, but too few hands to conclude. It is kept in `docs/data/early_ablation_seed100.txt` and is not part of the final numbers.

## 5. Slumbot

SLUMBOT_SECTION

## 6. Push-or-fold

`solve_pushfold(stack)` in `pokerbot.policy.pushfold`. The small blind may only go all-in or fold and the big blind may only call or fold. Every row is solved to an exploitability below 2e-6 big blinds.

| Stack, big blinds | Small blind goes all-in | Big blind calls | Small blind's value, big blinds per hand |
|---|---|---|---|
| 2 | 89.4% | 100.0% | +0.010 |
| 5 | 71.5% | 62.0% | +0.056 |
| 10 | 58.3% | 37.4% | -0.045 |
| 15 | 45.7% | 28.5% | -0.127 |
| 20 | 40.3% | 21.7% | -0.183 |
| 50 | 20.1% | 8.8% | -0.360 |

The value turns negative as stacks deepen because being restricted to all-in or fold becomes a handicap. That is why the bot uses this table only at 10 big blinds or less. Equities come from a Monte Carlo table with a standard error near 0.001 per entry, so the ranges are exact for that table, not for the true equities.

## 7. Speed

`uv run pokerbot bench`

| Measure | Value |
|---|---|
| Hand evaluator, one thread | 7 to 11 million 7-card hands per second |
| Equity of all 1326 combos against a range, all threads | flop 11 ms, turn 0.9 ms, river 0.1 ms |
| Same on the flop, one thread | 54 ms |
| Heuristic decision, median / 99th percentile | preflop 0.4 / 0.8 ms, flop 28 / 34 ms, turn 3 / 19 ms, river 1 / 3 ms |
| First decision in a fresh process | 160 ms |
| River solve, 200 iterations | 0.1 to 0.4 s depending on tree size |
| Heuristic self-play, one process | 21 hands per second |
| Peak memory | 270 MB |

Three optimizations made the evaluation affordable. None changed a result: after each one a reference match gave bit-identical numbers.

| Change | Effect |
|---|---|
| Rank-pair table per board and radix sort | Flop equity on one thread from about 200 ms to 54 ms |
| Compiled solver kernels | River solve about 3 times faster |
| One thread per match worker | Removed contention between worker processes |

## 8. What is not shown

- **Strength against strong opponents.** The only strong opponent available is Slumbot, and the sample there is small.
- **Whole-game exploitability.** Only the river subgame and push-or-fold are solved. The flop, turn and deep-stack preflop play are heuristic, and nothing here bounds how exploitable they are.
- **Small edges.** With about 13 hands per second for the solver bot, an edge under roughly 10 bb/100 could not be separated from zero in the time available.
- **Three or more players.** That layer is equity analysis and was not evaluated in matches.

## 9. What I would do next

In order of expected value, from the model-error result:

1. **Better range estimates.** Replace the fixed heuristic likelihood on the turn with solved strategies, by solving the turn to the end of the hand. Keep it only if a duplicate match shows a gain.
2. **Opponent adaptation.** Track how often an opponent folds, calls and raises, and tilt the assumed ranges, with a cap so the bot stays hard to exploit.
3. **Variance reduction.** Add all-in equity adjustment or AIVAT so that smaller edges become measurable.
