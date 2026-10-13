# Design

## The task

Build a bot that recommends the next best move in poker from the cards in front of you. The brief is one line, with no rules, interface or scoring given, so this document fixes those choices and gives the reasons. What was measured is in [RESULTS.md](RESULTS.md).

## Scope

**Game:** heads-up (two-player) No-Limit Texas Hold'em, no rake, objective is expected chips won.

**Why heads-up.** A two-player zero-sum game has a well-defined target. An equilibrium strategy guarantees at least the value of the game, averaged over both seats, against any opponent, and counterfactual regret minimization (CFR) converges to such a strategy. With three or more players equilibria still exist, but playing one gives no such guarantee and CFR has no convergence result. So heads-up is where a claim of "good play" can be checked, and that is where the rigorous part of this bot lives. Spots with three or more players get equity analysis and a conservative rule, and are labelled as such.

**Layers, in decreasing order of rigour.**

| Layer | Covers | Status |
|---|---|---|
| River solver | Heads-up river decisions | Solved with CFR; validated on small games and against an independent solver |
| Push-or-fold table | Button first to act with 10 big blinds or less | Solved exactly, as a restricted game |
| Heuristic advisor | Every other heads-up spot; all range updates | Rules derived from simple game theory, labelled heuristic |
| Multiway equity analysis | Three or more players | Equity plus a conservative rule; not a strategy |

**What is deliberately not here.** A full-game blueprint strategy or a value network. Published systems needed about 700 core-hours (Modicum) to 12,400 core-hours (Pluribus) for those. The time limit here is under three days on one laptop. Without them the flop and turn cannot be solved honestly, so they are not presented as solved.

## What the research supports

| Finding | Source | What this project takes from it |
|---|---|---|
| Libratus, DeepStack, Pluribus and ReBeL are all built on CFR plus real-time search | [Libratus](https://www.science.org/doi/10.1126/science.aao1733), [DeepStack](https://www.science.org/doi/10.1126/science.aam6960), [Pluribus](https://www.science.org/doi/10.1126/science.aay2400), [ReBeL](https://arxiv.org/abs/2007.13544) | CFR for the part that is solved. This does not show that a small CFR bot beats every other approach |
| Early streets in those systems rest on a trained blueprint or value network | [Depth-limited solving](https://arxiv.org/abs/1805.08195) | Early streets here are heuristic |
| From the turn onward Modicum solved to the end of the game with 150 to 2,000 CFR iterations, about 20 s per hand on 4 cores | [Depth-limited solving](https://arxiv.org/abs/1805.08195) | A solved river fits this machine; a solved turn might |
| Discounted CFR with alpha 1.5, beta 0, gamma 2 usually converges faster than CFR+ in poker subgames (an empirical result) | [Discounted CFR](https://arxiv.org/abs/1809.04040) | Plain CFR is validated first; DCFR is then benchmarked against it |
| LLM agents still lose to solver bots | [GTO Wizard Benchmark](https://arxiv.org/abs/2603.23660) | No LLM in the decision loop, also for latency, cost and reproducibility |
| Heads-up NLHE results have a standard deviation near 10 big blinds per hand | [AIVAT](https://arxiv.org/abs/1612.06915) | Only large edges are measurable here, so matches are duplicate pairs with intervals |
| A public reference river solver validated on Kuhn and Leduc | [noambrown/poker_solver](https://github.com/noambrown/poker_solver) | Same validation ladder; used as an external cross-check, not as a dependency |

## Contracts

These were fixed before any solver code. Every module follows them.

### Cards and hands

- A card is an integer `4 * rank + suit`, with rank 0 to 12 for 2 to Ace and suit 0 to 3 for clubs, diamonds, hearts, spades. Text form is rank then suit, for example `As`, `Td`, `2c`.
- A hole-card combo is an unordered pair `c1 < c2` with index `c2 * (c2 - 1) / 2 + c1`, from 0 to 1325.
- Hand strength is a single integer where larger is stronger and equal means a tie.

### Game state

- Chips are integers. The default game is blinds 50/100 with 100 big blinds each.
- Seat 0 is the button and small blind: it acts first before the flop and last after it. Seat 1 is the big blind.
- The state holds: blinds; for each seat its remaining stack, its contribution on this street and its total contribution to the pot; the seat to act; the size of the last raise increment; the street; the board; and the action history.
- Actions are fold, check, call, and bet or raise. **Every bet or raise is given as a "raise to" amount: the seat's total contribution on this street after the action.**
- The minimum raise is the current bet plus the last raise increment (at least one big blind), capped by the stack. Going all-in is always allowed when a raise is allowed.
- Heads-up needs no "re-opening" rule: a raise that is too small to be a full raise can only be an all-in, and the only replies to an all-in are call or fold.
- If one player is all-in and the other has called, the remaining board cards are dealt and the hand goes to showdown.
- One function, `legal_actions`, decides what is allowed. The engine, the view given to bots and the solver's bet tree all call it.

### Input modes

- **Complete state.** A full valid state gives an executable action that is guaranteed legal.
- **Partial input.** For example hole cards, board and pot only. The advisor fills the gaps with the simplest betting line that produces the stated pot (the button raised, the big blind called, earlier streets were checked) and lists every assumption it made.
- **Filled-in actions are not evidence.** The preflop line sets the starting ranges, because the pot size supports it. Checks invented for earlier streets are marked unobserved and never narrow a range. An early version did use them, and the model-error experiment in [RESULTS.md](RESULTS.md) is what exposed it.
- Invalid or contradictory input (a duplicated card, a pot that cannot be reached) returns a structured error with a stable code. It is never silently repaired.

### Payoffs

- A payoff is net chips from the start of the hand.
- At showdown the winner takes the pot; a tie splits it.
- Calling `C` into a pot of `P` (the pot includes the bet being called), with equity `e` and no betting afterwards, is worth `e * (P + C) - C` relative to folding.

### Solver values

- Values are kept per private hand: a vector with one entry per combo.
- The value of a hand at a node is the sum over opponent combos that share no card with it or with the board, weighted by the opponent's reach.
- A single range-against-range number is never used inside the solver, because it cannot give a bluff and the nuts different regrets.

### Ranges

- Each player's public range is a vector of 1326 weights.
- Ranges are updated after every action as it happens, using the board as it was at that moment.
- The hero's range stays a range inside the solver. The actual hole cards are used only to read off the recommendation at the end. Otherwise the modelled opponent would respond as if it knew the hero's hand.
- The update is Bayes' rule: `weight(hand) *= P(action | hand, public state)`.

### Opponent action likelihood

- `P(action | hand, public state)` comes from one fixed, versioned policy model: the heuristic advisor's own policy.
- It is mixed with a uniform component of weight 0.1, so an unexpected or off-tree action lowers the weight of surprising hands without removing them. A range can never collapse to zero.
- **Range floor.** When a range is read for a decision, 20% of it is the player's preflop range, whatever the postflop betting said. The per-action mixing above protects against one surprising action; the floor protects against the model being wrong about a whole line, for example an opponent who checks strong hands. The tracked range itself is still updated sharply, and the floor is applied only on reading, so it does not compound from street to street.

### Output

Every recommendation carries: the action and size; the hand's mixed strategy; the EV of each action where the layer can compute it; the hand's equity against the opponent's range; which layer answered; the assumptions made; the iterations completed; and the time taken. It is available as JSON and as readable text.

### Time limits

- A river solve runs in small batches with a clock check between batches. It always holds a valid strategy, and reports whether it stopped on accuracy, time or iterations.
- In matches the solver bot uses a fixed iteration budget instead of a time budget, so that the same seed gives the same result whatever else the machine is doing.
- Internal errors raise instead of being hidden by a fallback.

## Decision flow

1. Validate the input and pick the input mode.
2. **Three or more players:** equity against the opponents' ranges, by sampling compatible hands jointly and counting ties as fractional shares, then a conservative call, fold or raise rule. Side pots are rejected explicitly.
3. **Heads-up, before the flop:** see the next section.
4. **Heads-up, flop and turn:** the heuristic policy.
5. **Heads-up, river:** solve the subgame, with the real bet sizes in the tree.

After every action, on every street, both ranges are updated with the heuristic policy as the likelihood model.

## The heuristic policy

It always answers for a whole range at once: a matrix of hands by actions. One row is the hero's decision, and one column is the likelihood used to update a range. All its numbers live in one config object, `PolicyConfig`, which carries a version tag.

**Before the flop**

- Button, first to act, 10 big blinds or less: all-in or fold, following the solved push-or-fold equilibrium for that stack depth.
- Button, first to act, deeper: raise to 2.5 big blinds with the top 82% of hands ranked by equity against a random hand, fold the rest.
- Facing a raise: each hand is judged by its all-in equity against the raiser's current range, from a precomputed 169 x 169 table. Re-raise for value at 58% equity or more, plus a share of bluffs. Call when `equity * realization >= C / (P + C)`, where the realization factor (1.0 in position, 0.8 out of position) is an allowance for how much equity a call keeps through later betting.
- Calling an all-in ends the betting, so there the pot-odds rule is exact and no allowance is applied.

**After the flop**

Every hand is scored by its exact equity against the opponent's current range. The rules come from the solution of the simplest betting game, a polarized bettor against a bluff-catcher. For a bet of `s` times the pot:

- the caller must continue with `1 / (1 + s)` of its range (the minimum defence frequency), or any bluff shows a profit;
- the bettor may bluff `s / (1 + s)` hands for every value hand.

Applied here:

- **Bet for value** when equity is at least `1 - 1 / (2 (1 + s))`. If the opponent continues with the top `1 / (1 + s)` of its range, this is the point where the hand beats half of what continues.
- **Bluff** with weight `s / (1 + s)` per unit of value weight. On the river the bluffs are the hands with the least showdown value; earlier they are the weak hands with the most equity when called.
- **Call** when `e * (P + C) - C > 0` with a small margin, and **raise** the strongest hands with bluffs in the same ratio.
- **Never over-fold:** if the rules above would continue with less than 65% of the minimum defence frequency, the best of the folding hands are added to the calls.

These are heuristics. Equity on the flop and turn assumes the hand is checked to showdown, which misprices draws and position.

## How the river is solved

- **Inputs:** the board, the pot, both stacks, both ranges, and a bet-size configuration.
- **Tree:** starts at the actual situation, so a bet the opponent has made is in it exactly and no translation between sizes is needed. Below the root it keeps bets of 33%, 75% and 150% of the pot, raises of 75% of the pot after calling, and at most three bets and raises.
- **All-in limit.** All-in is offered only when it is at most three times the pot. A bet of many pots is right only if the range estimate is right nearly always, and the estimate does not deserve that trust. This is a risk limit, not a claim about equilibrium play.
- **Algorithm:** CFR in vector form: at each decision the strategy is a matrix of hands by actions. One traversal serves plain CFR, CFR+ and Discounted CFR. Discounted CFR is the default.
- **Terminal values in one pass.** At a fold every compatible opponent hand pays the same amount, so the value is that amount times the compatible reach, found from one total and 52 per-card sums. At showdown, hands are sorted by strength once per solve; the reach a hand beats is then a running sum, corrected by per-card running sums for the combos that share a card. Both are tested against the full 1326 x 1326 matrix product.
- **Quality measure:** exploitability, defined as half the sum of both players' best-response gains (NashConv / 2), as a percentage of the pot. It is exact for the given bet tree and root ranges and says nothing outside them.

The same solver code handles Kuhn poker, Leduc poker, the push-or-fold game and the river. Only the tree and the terminal values differ, so the checks on the small games are checks on the code that solves the river.

## Three kinds of error

The write-up keeps these apart, because a good number on one says nothing about the others.

| Kind | Question | How it is measured |
|---|---|---|
| Numerical convergence | Is the subgame solved accurately? | Exploitability within the bet tree and root ranges |
| Model error | Is it the right subgame? | How the recommendation moves under other range assumptions and with extra bet sizes |
| Empirical performance | Does it win chips? | Duplicate matches, bb/100 with 95% confidence intervals |

## Making it fast enough

Speed sets how many hands an evaluation can afford. Three changes mattered, and none changes a result:

- **Rank-pair table.** On a fixed five-card board, a hand that makes no flush is decided by the ranks of its two hole cards. So 91 rank pairs are evaluated once per board and all 1326 combos become lookups; only combos that complete a flush are worked out one by one.
- **Radix sort.** Strengths fit in 24 bits, so sorting about a thousand combos takes three counting passes instead of a comparison sort.
- **Compiled solver kernels.** Profiling showed most solver time was NumPy call overhead on small arrays, not arithmetic. Regret matching and the regret updates are two small Numba functions.

## Verification

- **Evaluator:** category counts over all 2,598,960 five-card and all 133,784,560 seven-card hands; ordering cases inside each category; invariance to card order and suit relabelling; a 7-card result equals its best 5-card subset; agreement with `phevaluator`.
- **Equity:** the fast range method equals a direct all-pairs computation; exact small cases; Monte Carlo checks with statistical tolerances.
- **Rules engine:** scripted hands for minimum raise, short all-in and split pots; chip conservation at every step of random playouts; a bot never receives hidden cards.
- **CFR:** Kuhn poker reaches -1/18 with the known equilibrium structure; Leduc reaches the published value.
- **River solver:** fast terminal values equal the matrix product; the bet tree agrees with the engine on random lines; a polarized toy game reproduces the theoretical bluffing frequency at three bet sizes; an independent solver's best-response code reports the same exploitability.
- **Push-or-fold:** solved to an exploitability near zero; the 10 big blind ranges match the published chart.
- **Matches:** duplicate pairs, intervals over pairs, tuning seeds separate from evaluation seeds, sample sizes chosen from measured throughput.

## Known limits

- Before the flop, on the flop and on the turn the bot is heuristic. Only the river and short-stack push-or-fold are solved, each within its own stated game.
- The river solve depends on the ranges it is given, and those come from the heuristic model. [RESULTS.md](RESULTS.md) shows this is the largest source of error, and that against Slumbot it led to all-in bets of many times the pot before the range floor and the all-in limit were added.
- The two guards cost value against opponents who never fold, since unlimited value bets are ideal there.
- Solving each spot well does not bound the exploitability of the whole strategy.
- The bot does not adapt to its opponent. Against a bot that never folds it still bluffs, and it does not lift its all-in limit.
- Support for three or more players is equity analysis with a conservative rule.
- Small edges cannot be measured in the time available.

## Design review

The first version of this design was reviewed by a second model (GPT-6 Astra through Codex, read-only) before any code was written. Its verdict was "I would not build this plan as written." The main changes that followed:

| Finding | Change |
|---|---|
| Solving the flop and turn with showdown equity at the leaves solves the wrong game, and was wrongly credited to Modicum | Flop and turn are heuristic and are not presented as solved |
| Payoff, range and legality rules were underspecified | The Contracts section above |
| Replaying history through the bot's own solver is a fragile opponent model | Incremental range updates, a fixed likelihood model, a uniform component |
| Exploitability claims were too broad | Claims scoped to the bet tree and root ranges; model error measured separately |
| A preflop solve with equity-realization factors is not constant-sum | The preflop policy is a labelled heuristic; only push-or-fold is solved |
| Sample sizes could not show small edges | Benchmark first, intervals over duplicate pairs |
| Catch-all fallbacks hide bugs | Bounded batches, a fixed iteration budget in matches, loud failures |
