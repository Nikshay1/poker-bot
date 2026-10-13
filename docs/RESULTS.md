# Results

Everything here was measured on one laptop (16 threads, 15 GB RAM). Each number can be reproduced with the command given next to it. The raw data is in [`docs/data/`](data).

## Summary

1. **The solver is correct.** It reproduces the known solutions of Kuhn and Leduc poker, matches the published push-or-fold chart, and an independent solver's best-response code reports the same exploitability for its river solutions to nine decimal places.
2. **The river solver earns chips.** Adding it to the heuristic bot wins **+33.7 bb/100** (95% interval +22.6 to +44.7) over 20,000 hands in a head-to-head duplicate match.
3. **Solver precision is not the limit; the assumptions are.** A river solve reaches about 0.1% of the pot in exploitability. Changing the assumed opponent range still changes the recommended action in up to 16% of test situations.
4. **Two mistakes were found by measurement and fixed.** The model-error experiment exposed that made-up history was being used as evidence. A session against Slumbot exposed all-in bets of up to 21 times the pot. Both came from trusting the range estimate too much.
5. **Both bots beat all three baselines** by wide margins, with no illegal action in 60,000 evaluation hands.
6. **Strength against strong opponents is not established.** 300 hands against Slumbot gave +135 bb/100 with an interval of -67 to +337.

Terms used below:

- **bb/100**: big blinds won per 100 hands.
- **Duplicate pair**: the same cards played twice with the seats swapped, so card luck largely cancels. Intervals are computed over pairs.
- **Exploitability**: how much a perfect opponent would gain against a strategy, averaged over both seats. Zero means an equilibrium. Here it is always measured inside a stated game: a given bet tree and given starting ranges.

## 1. Correctness

`uv run pytest` runs 126 checks in about 20 seconds; `uv run pytest -m slow` adds the exhaustive one.

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

Exploitability as a percentage of the pot, on four river situations with tracked ranges and the default bet tree. Each cell lists the four situations in order: first to act (22 decisions), facing a bet (5), checked to (11), first to act on a paired board (26).

| Iterations | Plain CFR | CFR+ | Discounted CFR | DCFR time, seconds |
|---|---|---|---|---|
| 50 | 5.55 / 1.23 / 1.31 / 4.28 | 2.16 / 0.26 / 0.50 / 1.37 | 0.755 / 0.082 / 0.557 / 0.865 | 0.09 / 0.02 / 0.05 / 0.11 |
| 100 | 2.99 / 0.58 / 0.82 / 2.07 | 0.66 / 0.09 / 0.19 / 0.59 | 0.266 / 0.028 / 0.163 / 0.438 | 0.19 / 0.04 / 0.09 / 0.22 |
| 200 | 1.40 / 0.27 / 0.43 / 1.07 | 0.26 / 0.04 / 0.07 / 0.27 | 0.097 / 0.010 / 0.065 / 0.158 | 0.37 / 0.08 / 0.19 / 0.44 |
| 400 | 0.70 / 0.14 / 0.24 / 0.56 | 0.11 / 0.02 / 0.03 / 0.11 | 0.036 / 0.005 / 0.020 / 0.052 | 0.75 / 0.17 / 0.37 / 0.88 |
| 800 | 0.36 / 0.07 / 0.13 / 0.29 | 0.05 / 0.01 / 0.01 / 0.04 | 0.011 / 0.001 / 0.006 / 0.015 | 1.50 / 0.33 / 0.75 / 1.76 |

What this shows:

- Discounted CFR is the most accurate from 100 iterations on, in all four situations. At 800 iterations it is 19 to 64 times more accurate than plain CFR.
- A solve of 200 iterations takes 0.1 to 0.4 seconds and reaches 0.01% to 0.16% of the pot. The match bot stops at 0.25% of the pot or 300 iterations.
- Bigger trees converge more slowly and cost more per iteration.

On the small games the same pattern holds. After 1,000 iterations on Leduc poker, exploitability is 0.0118 for plain CFR, 0.00026 for CFR+ and 0.00017 for Discounted CFR, in units of the ante.

## 3. Model error

`uv run python scripts/model_error.py`

A solve can be exact and still answer the wrong question, because the opponent's range and the bet sizes are assumptions. This experiment holds 64 river situations fixed (4 boards, 4 hands, 4 betting lines) and changes one assumption at a time. Every solve reached about 0.08% of the pot, so the differences below are not numerical noise.

| Assumption changed | Top action changes | Distance, hero's hand | Distance, hero's whole range |
|---|---|---|---|
| Range inferred less sharply (uniform component 0.3 instead of 0.1) | 2 of 64 (3%) | 0.04 | 0.04 |
| Opponent may hold any hand | 10 of 64 (16%) | 0.17 | 0.15 |
| Richer bet tree (4 bet sizes, 2 raise sizes, unlimited all-in) | 1 of 64 (2%) | 0.03 | 0.03 |

Distance is total-variation distance between strategies, grouped into fold, check or call, and bet or raise: 0 means identical, 1 means completely different.

An example. With King-Queen on a board of 7h 8h 2c Kd 3s, facing a bet of two thirds of the pot:

| Opponent range | Solver's answer for King-Queen |
|---|---|
| The model's estimate, which accounts for the bet | Call, 99% |
| Any hand | Raise, 100% |

Both answers are numerically exact for their own inputs. Ignoring what the bet says about the opponent's hand turns a call into a raise.

### This experiment found a design mistake

The first run of this experiment gave much larger numbers:

| Assumption changed | First run | After the fix | Final version |
|---|---|---|---|
| Range inferred less sharply | 13 of 64 (20%) | 3 of 64 (5%) | 2 of 64 (3%) |
| Opponent may hold any hand | 19 of 64 (30%) | 10 of 64 (16%) | 10 of 64 (16%) |
| Richer bet tree | 2 of 64 (3%) | 4 of 64 (6%) | 1 of 64 (2%) |

The cause: these situations are built from partial input (cards and pot), and the advisor fills in the missing history as "earlier streets were checked through". The range tracker then used those made-up checks as evidence that the opponent was weak. The clearest symptom was pocket aces on 9s 8s 7d 6c 2h facing a bet, where the solver raised 100% of the time. In a similar spot it recommended an all-in with one pair.

The fix: filled-in actions are marked as unobserved and no longer narrow any range. The aces now call. Match play was never affected, because a match has a complete, real history; a re-run evaluation match gave a bit-identical result. The first-run data is kept in `docs/data/model_error_before_fix.json`. The final column also includes the two guards from section 5, and its richer tree is larger than in the earlier runs.

What follows from this:

- The solver's 0.1% figure must not be read as the accuracy of the bot's decisions. The range assumption moves the answer far more often than numerical error could.
- Where the history is real, inference is only as good as the model of the opponent. Section 5 shows what happens when that model is wrong.

## 4. Match results

`uv run python scripts/evaluate.py --seed 2000` then `uv run python scripts/report.py docs/data/matches_seed2000.json`

Final evaluation: seed 2000, used for nothing else, with the bots frozen at commit `d9bfd00`. Stacks are 100 big blinds, reset every hand.

| Bot | Opponent | Win rate, bb/100 | 95% interval | Hands |
|---|---|---|---|---|
| solver | heuristic | +33.7 | +22.6 to +44.7 | 20,000 |
| solver | sharp | -7.1 | -30.8 to +16.5 | 10,000 |
| solver | equity | +170.0 | +131.7 to +208.4 | 6,000 |
| solver | call | +334.1 | +275.4 to +392.8 | 3,000 |
| solver | random | +293.7 | +225.9 to +361.5 | 3,000 |
| heuristic | equity | +150.5 | +116.5 to +184.6 | 6,000 |
| heuristic | call | +276.7 | +238.7 to +314.7 | 3,000 |
| heuristic | random | +293.0 | +223.1 to +362.9 | 3,000 |
| equity | call | +201.5 | +183.2 to +219.7 | 3,000 |
| equity | random | +605.1 | +525.3 to +684.9 | 3,000 |

Illegal actions across all 60,000 hands: 0.

The bots:

- **solver**: the heuristic bot with river decisions solved. This is the final bot.
- **heuristic**: the heuristic advisor on every street.
- **sharp**: the solver bot without the two guards of section 5.
- **equity**: plays its equity against a random hand, ignoring what the betting says. This is the natural first attempt at the task.
- **call**: never folds, never raises.
- **random**: picks among fold, check or call, and a random raise.

How to read the table:

- **The first row is the ablation.** The two bots differ only on the river, so +33.7 bb/100 is what the river solver adds. The interval excludes zero by a wide margin.
- **The second row is the price of the guards.** Head to head against the unguarded version the difference is -7.1 with an interval from -30.8 to +16.5: not separated from zero.
- The heuristic bot beats the equity bot by 150 bb/100. Reading the opponent's range from the betting is worth more than raw hand strength.
- These baselines are weak. Beating them shows the bots are sound, not that they are strong. Neither bot adapts to its opponent: the equity bot beats the random bot by more than the solver does, because a bot that simply bets its good hands happens to exploit random play well.

## 5. Slumbot, and two guards against overconfidence

`uv run pokerbot slumbot --bot solver --hands 300 --log <file>`, then `scripts/slumbot_report.py` and `scripts/replay_overbets.py` on the log.

Slumbot is the 2018 Annual Computer Poker Competition champion, playable over a public API at 200 big blinds deep. These hands are not duplicate pairs, so the intervals are over single hands and are very wide.

### What the first session showed

| | Before the guards (`sharp`) | Final bot |
|---|---|---|
| Hands | 300 | 300 |
| Win rate, bb/100 | +10.3 | +134.8 |
| 95% interval | -215.9 to +236.6 | -67.4 to +336.9 |
| Bets of 100 big blinds or more | 13 | 2 |
| Bets after the flop of more than twice the pot | 12 | 1 |
| Hands won or lost for 150 big blinds or more | 2 | 0 |

**Neither win rate says anything about strength.** Both intervals include zero comfortably, and the difference between the two sessions is also noise.

The betting profile is what mattered. In the first session the bot moved all-in 13 times, in most cases for 6 to 21 times the pot. In one hand it held Jack-Two on a board of Ks 8d 7h 9h 6c, Slumbot checked for the third time, and the bot bet 192 big blinds into a pot of 15 as a bluff. Slumbot held a straight.

The cause is the same as in section 3, this time with real history. The range model assumes the opponent plays like the bot's own heuristic, which bets its strong hands. After three checks the model had removed every strong hand from Slumbot's range, and against a range with no strong hands a huge bet is correct. A strong opponent checks strong hands some of the time.

### The two guards

- **Range floor.** When a range is read for a decision, 20% of it is the player's preflop range, whatever the postflop betting said. This stands for "my model of this opponent may be wrong about the whole line". The tracked range is still updated sharply; the floor is applied only on reading, so it does not compound.
- **All-in limit.** The bet tree offers all-in only when it is at most three times the pot, and gains a bet of 1.5 times the pot so that large bets remain possible. This is a hard risk limit: a bluff of 20 pots is only right if the opponent folds more than 95% of the time, and nothing here justifies that confidence.

### Evidence for adopting them

Replaying the 13 logged spots. For each, the situation was rebuilt exactly and each variant was asked what it would do. The number is the share of the hand's strategy that is all-in.

| Variant | Average all-in share over the 13 spots | The Jack-Two bluff |
|---|---|---|
| No guards | 90% | 89% |
| Range floor 0.1 | 54% | 71% |
| Range floor 0.2 | 40% | 1% |
| Range floor 0.2 and all-in limit (final) | 8% | 0% |

The one spot where the final bot still moves all-in is a made flush betting 1.9 times the pot.

Tuning matches on seed 200, each 6,000 hands unless noted. "Trapper" is the unguarded solver bot changed to check its strongest hands on the flop and turn, which is exactly the behaviour the range model does not expect.

| Variant | Against the unguarded bot | Against the equity bot (4,000 hands) | Against the trapper |
|---|---|---|---|
| No guards | | +191.8 (+125.1 to +258.5) | +50.6 (0.0 to +101.3) |
| Range floor 0.1 | -14.3 (-42.8 to +14.2) | | |
| Range floor 0.2 | -24.1 (-55.4 to +7.3) | +186.0 (+128.5 to +243.5) | +77.0 (+28.0 to +125.9) |
| All-in limit only | -9.2 (-29.2 to +10.9) | | |
| Both (final) | -17.3 (-47.5 to +12.9) | | +86.9 (+38.9 to +134.9) |

How to read this:

- No comparison in the table is statistically significant. The guards were not adopted because they win more chips.
- They were adopted because they remove a failure that was seen in real play, at a cost that could not be distinguished from zero in three tuning matches and again on the fresh evaluation seed (-7.1, -30.8 to +16.5).
- The direction against the trapper favours the guards, as expected, but that is a hint and not a result.
- **The guards do have a known cost.** Against the always-call bot the final bot wins +334.1 bb/100 (+275.4 to +392.8) and the unguarded bot +488.7 (+346.9 to +630.4) on the same seed. Unlimited all-in value bets are ideal against an opponent that never folds. A bot that adapted to its opponent would lift the limit there; this one does not adapt.

## 6. Tuning record

All tuning used seed 200. The evaluation seeds (1000, then 2000) were not used for tuning.

| Change | Evidence | Decision |
|---|---|---|
| Bluff at the theoretical ratio on every street, instead of 2x on the flop and 1.5x on the turn | +53.8 (+22.1 to +85.5) head to head; against the equity bot +154.7 instead of +101.2 | Adopted |
| Infer ranges less sharply (uniform component 0.3) | -48.2 (-82.1 to -14.3) head to head; no change against the equity bot | Rejected. The head-to-head loss is weak evidence, because there the opponent plays exactly the model that sharp inference assumes |
| Range floor and all-in limit | Section 5 | Adopted as risk controls |

An evaluation of the version before the guards was run on seed 1000 and is kept in `docs/data/matches_seed1000.json`. There the solver bot beat the heuristic bot by +39.1 (+26.4 to +51.9) over 20,000 hands, in line with the final figure.

## 7. Push-or-fold

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

## 8. Speed

`uv run pokerbot bench`

| Measure | Value |
|---|---|
| Hand evaluator, one thread | 7 to 11 million 7-card hands per second |
| Equity of all 1326 combos against a range, all threads | flop 11 ms, turn 0.9 ms, river 0.1 ms |
| Same on the flop, one thread | 54 ms |
| Heuristic decision, median / 99th percentile | preflop 0.4 / 0.6 ms, flop 28 / 35 ms, turn 3 / 20 ms, river 1 / 3 ms |
| First decision in a fresh process | 160 ms |
| River solve, 200 iterations | 0.1 to 0.4 s depending on tree size |
| Heuristic self-play, one process | 21 hands per second |
| Solver bot in a 16-process match | about 15 hands per second |
| Peak memory | 270 MB |

Three optimizations made the evaluation affordable. None changed a result: after each one a reference match gave bit-identical numbers.

| Change | Effect |
|---|---|
| Rank-pair table per board and radix sort | Flop equity on one thread from about 200 ms to 54 ms |
| Compiled solver kernels | River solve about 3 times faster |
| One thread per match worker | Removed contention between worker processes |

## 9. What is not shown

- **Strength against strong opponents.** The only strong opponent available is Slumbot, and 300 hands cannot separate the bot from it in either direction.
- **Whole-game exploitability.** Only the river subgame and push-or-fold are solved. The flop, turn and deep-stack preflop play are heuristic, and nothing here bounds how exploitable they are.
- **Small edges.** At about 15 hands per second, an edge under roughly 10 bb/100 could not be separated from zero in the time available.
- **Three or more players.** That layer is equity analysis and was not evaluated in matches.
- **Robustness beyond the two failures found.** The guards address what was observed. Other consequences of a wrong opponent model may remain.

## 10. What I would do next

In order of expected value:

1. **Better range estimates.** Replace the fixed heuristic likelihood on the turn with solved strategies, by solving the turn to the end of the hand. Keep it only if a duplicate match shows a gain.
2. **Opponent adaptation.** Track how often an opponent folds, calls and raises, and adjust the assumed ranges and the all-in limit, with a cap so the bot stays hard to exploit. The always-call result shows what this is worth.
3. **Variance reduction.** Add all-in equity adjustment or AIVAT so that smaller edges, and a Slumbot result, become measurable.
