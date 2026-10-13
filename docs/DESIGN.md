# Design

## The task

Build a bot that recommends the next best move in poker from the cards in front of you. The brief is one line, with no rules, interface or scoring given, so this document fixes those choices and gives the reasons.

## Scope

**Game:** heads-up (two-player) No-Limit Texas Hold'em, no rake, objective is expected chips won.

**Why heads-up.** A two-player zero-sum game has a well-defined target. An equilibrium strategy guarantees at least the value of the game, averaged over both seats, against any opponent, and counterfactual regret minimization (CFR) converges to such a strategy. With three or more players equilibria still exist, but playing one gives no such guarantee and CFR has no convergence result. So heads-up is where a claim of "good play" can be checked, and that is where the rigorous part of this bot lives. Spots with three or more players get equity analysis and a conservative rule, and are labelled as such.

**Three layers, in decreasing order of rigour.**

| Layer | Covers | Status |
|---|---|---|
| Heuristic advisor | Every heads-up spot on every street; equity analysis for 3+ players | Always available, labelled heuristic |
| River solver | Heads-up river decisions | Solved with CFR, validated |
| Experimental | Turn solved to the end of the hand, and others | Used only if a measured ablation shows it helps |

**What is deliberately not here.** A full-game blueprint strategy or a value network. Published systems needed about 700 core-hours (Modicum) to 12,400 core-hours (Pluribus) for those. The time limit here is under three days on one laptop.

## What the research supports

| Finding | Source | What this project takes from it |
|---|---|---|
| Libratus, DeepStack, Pluribus and ReBeL are all built on CFR plus real-time search | [Libratus](https://www.science.org/doi/10.1126/science.aao1733), [DeepStack](https://www.science.org/doi/10.1126/science.aam6960), [Pluribus](https://www.science.org/doi/10.1126/science.aay2400), [ReBeL](https://arxiv.org/abs/2007.13544) | CFR for the part that is solved. This does not show that a small CFR bot beats every other approach |
| Early streets in those systems rest on a trained blueprint or value network | [Depth-limited solving](https://arxiv.org/abs/1805.08195) | Early streets here are heuristic |
| From the turn onward Modicum solved to the end of the game with 150 to 2,000 CFR iterations, about 20 s per hand on 4 cores | [Depth-limited solving](https://arxiv.org/abs/1805.08195) | A solved river fits this machine; a solved turn might |
| Discounted CFR with alpha 1.5, beta 0, gamma 2 usually converges faster than CFR+ in poker subgames (an empirical result) | [Discounted CFR](https://arxiv.org/abs/1809.04040) | Plain CFR is validated first; DCFR is then benchmarked against it |
| LLM agents still lose to solver bots | [GTO Wizard Benchmark](https://arxiv.org/abs/2603.23660) | No LLM in the decision loop, also for latency, cost and reproducibility |
| Heads-up NLHE results have a standard deviation near 10 big blinds per hand | [AIVAT](https://arxiv.org/abs/1612.06915) | 20,000 hands give roughly ±14 bb/100, so only large edges are measurable here |
| A public reference river solver validated on Kuhn and Leduc poker | [noambrown/poker_solver](https://github.com/noambrown/poker_solver) | Same validation ladder; used as an external cross-check, not as a dependency |

## Contracts

These are fixed before any solver code. Every module follows them.

### Cards and hands

- A card is an integer `4 * rank + suit`, with rank 0 to 12 for 2 to Ace and suit 0 to 3 for clubs, diamonds, hearts, spades. Text form is rank then suit, for example `As`, `Td`, `2c`.
- A hole-card combo is an unordered pair `c1 < c2` with index `c2 * (c2 - 1) / 2 + c1`, from 0 to 1325.
- Hand strength is a single integer where larger is stronger and equal means a tie.

### Game state

- Chips are integers. The default game is blinds 50/100 (the convention Slumbot uses).
- Seat 0 is the button and small blind: it acts first before the flop and last after it. Seat 1 is the big blind.
- The state holds: blinds; for each seat its remaining stack, its contribution on this street and its total contribution to the pot; the seat to act; the size of the last raise increment; the street; the board; and the action history.
- Actions are fold, check, call, and bet or raise. **Every bet or raise is given as a "raise to" amount: the seat's total contribution on this street after the action.**
- The minimum raise is the current bet plus the last raise increment (at least one big blind), capped by the stack. Going all-in is always allowed when a raise is allowed.
- Heads-up needs no "re-opening" rule: a raise that is too small to be a full raise can only be an all-in, and the only replies to an all-in are call or fold.
- If one player is all-in and the other has called, the remaining board cards are dealt and the hand goes to showdown.

### Input modes

- **Complete state.** A full valid state gives an executable action that is guaranteed legal.
- **Partial input.** For example hole cards, board and pot only. The advisor answers for a named hypothetical scenario and lists every assumption it made.
- Invalid or contradictory input (a duplicated card, a call amount larger than the stack) returns a structured error. It is never silently repaired.

### Payoffs

- A payoff is net chips from the start of the hand.
- At showdown the winner takes the pot; a tie splits it.
- Calling `C` into a pot of `P` (the pot includes the bet being called), with equity `e` and no betting afterwards, is worth `e * (P + C) - C` relative to folding.

### Solver values

- Values are kept per private hand: a vector with one entry per combo in the player's range.
- The value of a hand at a node is the sum over opponent combos that share no card with it or with the board, weighted by the opponent's counterfactual reach.
- A single range-against-range number is never used inside the solver, because it cannot give a bluff and the nuts different regrets.

### Ranges

- Each player's public range is a vector of 1326 weights.
- Ranges are updated after every action as it happens, using the board as it was at that moment.
- The hero's range stays a range inside the solver. The actual hole cards are used only to read off the recommendation at the end. Otherwise the modelled opponent would respond as if it knew the hero's hand.
- The update is Bayes' rule: `weight(hand) *= P(action | hand, public state)`.

### Opponent action likelihood

- `P(action | hand, public state)` comes from one fixed, versioned policy model (the heuristic advisor's own policy).
- It is mixed with a small uniform component, so an unexpected or off-tree action lowers the weight of surprising hands without removing them. A range can never collapse to zero.

### Output

Every recommendation carries: the action and size; the mixed-strategy probabilities; the EV of each action where the layer can compute it; the hand's equity against the opponent's range; which layer answered; the assumptions made; the iterations completed; and the time taken. It is available as JSON and as readable text.

### Time limits

- Solving runs in bounded batches of iterations with a clock check between batches, and always holds a valid strategy.
- On timeout the best strategy so far is returned and the reason is recorded.
- The rate of fallbacks is logged. In development, internal errors raise instead of falling back.

## Decision flow

1. Validate the input and pick the input mode.
2. **Three or more players:** equity against the opponents' ranges, by sampling compatible hands jointly and counting ties as fractional shares, then a conservative call, fold or raise rule. Side pots are rejected explicitly if unsupported.
3. **Heads-up, before the flop:** a threshold chart over a hand ordering, with parameters per stack depth. The push/fold table is used only in the game it was solved for: small blind first to act and choosing between all-in and fold, big blind choosing between call and fold.
4. **Heads-up, flop and turn:** a range-aware rule. Call when `e * (P + C) - C > 0` against the estimated betting range, guarded by the minimum defence frequency. Bet for value when ahead of the estimated calling range. Bluff a share of weak draws that depends on the bet size.
5. **Heads-up, river:** solve the subgame, with the real bet sizes in the tree.

## How the river is solved

- **Inputs:** the board, the pot, both stacks, both ranges, and a bet-size configuration (pot fractions plus all-in, with a cap on the number of raises).
- **Tree:** built from the bet-size configuration. A bet the opponent actually made is always added to the tree, so no translation between sizes is needed.
- **Algorithm:** CFR in vector form. At each node the strategy is a matrix of hands by actions. Plain CFR is the reference; Discounted CFR is the default once benchmarked.
- **Terminal values:** at a fold, the folder loses what it put in. At showdown, hands are sorted by strength once, and each hand's value comes from prefix sums of the opponent's reach, corrected for combos that share a card. This is checked against a direct all-pairs computation.
- **Quality measure:** exploitability, defined as half the sum of both players' best-response gains (NashConv divided by 2), as a percentage of the pot. It is exact for the given bet tree and root ranges and says nothing outside them.

## Three kinds of error

The write-up keeps these apart, because a good number on one says nothing about the others.

| Kind | Question | How it is measured |
|---|---|---|
| Numerical convergence | Is the subgame solved accurately? | Exploitability within the bet tree and root ranges |
| Model error | Is it the right subgame? | How the recommendation moves under other range priors and with extra bet sizes |
| Empirical performance | Does it win chips? | Duplicate matches, bb/100 with 95% confidence intervals |

## Verification

- **Evaluator:** category counts over all 2,598,960 five-card hands; ordering cases inside each category; invariance to card order and suit relabelling; differential test against `phevaluator`.
- **Equity:** exact small cases with fixed suits; Monte Carlo checks with statistical tolerances.
- **Rules engine:** scripted hands for minimum raise, short all-in and split pots; chip conservation on random playouts; a test that a bot cannot read hidden cards.
- **CFR:** Kuhn poker value of -1/18 for the first player with exploitability near zero for both players; Leduc exploitability curve.
- **River solver:** fast terminal values equal the all-pairs computation; exploitability reported with its scope; one subgame cross-checked against the public reference solver.
- **Matches:** bb/100 with confidence intervals computed over duplicate pairs; tuning seeds separate from evaluation seeds; sample sizes chosen from measured throughput.

## Known limits

- Before the flop, on the flop and on the turn the bot is heuristic. Only the river is solved, and only within its bet tree and the ranges assumed at its root.
- Solving each spot well does not bound the exploitability of the whole strategy.
- Support for three or more players is equity analysis with a conservative rule.
- Small edges cannot be measured in the time available.

## Design review

The first version of this design was reviewed by a second model (GPT-6 Astra through Codex, read-only) before any code was written. Its verdict was "I would not build this plan as written." The main changes that followed:

| Finding | Change |
|---|---|
| Solving the flop and turn with showdown equity at the leaves solves the wrong game, and was wrongly credited to Modicum | Flop and turn are heuristic; that solve survives only as an ablation |
| Payoff, range and legality rules were underspecified | The Contracts section above |
| Replaying history through the bot's own solver is a fragile opponent model | Incremental range updates, a fixed likelihood model, a uniform component |
| Exploitability claims were too broad | Claims scoped to the bet tree and root ranges; model-error tests added |
| A preflop solve with equity-realization factors is not constant-sum | The preflop policy is a labelled heuristic chart |
| Sample sizes could not show small edges | Benchmark first, intervals over duplicate pairs |
| Catch-all fallbacks hide bugs | Bounded batches, logged fallback rate, loud failures in development |
