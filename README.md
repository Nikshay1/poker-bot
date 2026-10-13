# pokerbot

A decision bot for heads-up No-Limit Texas Hold'em. Given the cards and the betting situation, it recommends the next move and says how it got there.

```
$ uv run pokerbot advise --hole AsAd --board 9s8s7d6c2h --pot 1000 --to-call 400

Hand AsAd on 9s8s7d6c2h (river, button)
Pot 1000 (10.0 bb), to call 400, stack behind 9700

Recommended: call 400
Strategy for this hand: call 400 100%
Equity against the opponent's estimated range: 38.0%
EV in chips, relative to folding: fold +0, call 400 +133, raise to 1450 -233
Answered by: river solver layer in 122 ms
Assumptions:
  - heads-up, no rake
  - both players started the hand with 100 big blinds
  - before the flop the button raised to 300 and the big blind called
  - earlier streets were checked through; this is filler and is not used to narrow ranges
  - the big blind bet 400 on this street
Notes:
  - solved 35 iterations and reached the accuracy target
  - exploitability 0.15% of the pot, within this bet tree and these ranges
  - bet tree: 5 decisions, 10 endings
```

## How it works

| Layer | Covers | How |
|---|---|---|
| River solver | Heads-up river decisions | Counterfactual regret minimization (CFR) over both players' ranges |
| Push-or-fold table | Button with 10 big blinds or less | The same solver on the all-in-or-fold game |
| Heuristic advisor | Every other heads-up spot | Rules from simple game theory, applied to exact equity against the opponent's range |
| Multiway analysis | Three or more players | Monte Carlo equity and a cautious rule |

Throughout a hand the bot keeps a range for each player and updates both after every action. The river solver is given those ranges.

The reasoning behind each choice is in [docs/DESIGN.md](docs/DESIGN.md). Everything that was measured is in [docs/RESULTS.md](docs/RESULTS.md).

## Results in brief

- **Correct:** the solver reproduces the known solutions of Kuhn and Leduc poker and the published push-or-fold chart. An independent solver's best-response code reports the same exploitability for its river solutions, to nine decimal places.
- **Useful:** adding the river solver to the heuristic bot wins +33.7 bb/100 (95% interval +22.6 to +44.7) over 20,000 hands.
- **Honest about its limit:** a river solve is accurate to about 0.1% of the pot, but the recommended action changes in up to 16% of test situations when the assumed opponent range changes. The assumptions matter more than the arithmetic.
- **Corrected by measurement:** a session against Slumbot showed the bot staking whole stacks on small pots because it trusted its read of the opponent too much. Two guards now prevent that, at no cost that could be measured head to head.
- **Sound:** both bots beat a random bot, an always-call bot and an equity-only bot by wide margins, with no illegal action in 60,000 hands.
- **Not shown:** strength against strong opponents. 300 hands against Slumbot are far too few to tell.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest            # 143 checks, about 25 seconds
uv run pytest -m slow    # adds the exhaustive check of all 133,784,560 seven-card hands
```

## Use

**Follow a hand in the browser.** This is the way to use it while a hand is being played.

```
uv run pokerbot ui                    # opens http://localhost:8765/
uv run pokerbot ui --host 0.0.0.0     # also reachable from a phone on the same network
```

- Set the blinds, both stacks and your seat once. Pick your two cards.
- Enter each action as it happens, and the board cards when they are dealt. Whenever it is your turn the page shows the recommended action, how the bot would mix its play, your equity and the EV of each option.
- Record what you actually did, which may differ from the advice. One button records the recommended action.
- When a hand ends, the stacks and the button carry over to the next one. After a showdown you say who won.
- Cards can also be typed: rank then suit, for example `A` `S` `K` `D`. Backspace undoes the last entry.
- The **3+ players** tab is a one-shot form (cards, pot, amount to call, players). It gives the multiway equity analysis, which is not a strategy.

The heads-up tab assumes two players were dealt in. It does not model antes or straddles. The page runs on your machine and needs no internet connection.

**Advice for one situation.** Only the hole cards are required. Anything left out is filled with a stated assumption.

```
uv run pokerbot advise --hole AsKd
uv run pokerbot advise --hole 9h6h --board 7h8h2c --pot 1000 --to-call 400
uv run pokerbot advise --hole QsQd --position bb --pot 350 --to-call 150 --json
uv run pokerbot advise --hole AsAd --board 7h8h2c --pot 1000 --to-call 400 --players 4
```

- Chips are in units where the big blind is 100, unless `--bb` says otherwise.
- `--pot` includes any bet you are facing.
- On the river the answer comes from the solver, with up to `--time` seconds (default 2).

**A match between two bots**, played as duplicate pairs. Bots: `random`, `call`, `equity`, `heuristic`, `solver`, and `sharp` (the solver bot without its two guards).

```
uv run pokerbot match solver heuristic --pairs 1000 --workers 16
uv run pokerbot match heuristic:epsilon=0.3 heuristic --pairs 1000 --workers 16
```

The second form overrides one policy value, to compare variants without editing code.

**Other commands**

```
uv run pokerbot bench                        # speed and latency on this machine
uv run pokerbot slumbot --hands 100          # play Slumbot over its public API
uv run python scripts/show_hands.py solver equity --hands 10   # read hands action by action
```

**From Python**

```python
from pokerbot.advisor import SolverConfig, advise, spot_from_partial
from pokerbot.cards import parse_cards

spot = spot_from_partial(parse_cards("AsKd"), parse_cards("7h8h2c"), pot=1000, to_call=400)
rec = advise(spot, solver=SolverConfig())
print(rec.action, rec.probabilities, rec.assumptions)
```

## Layout

```
src/pokerbot/
  cards.py, evaluator.py, equity.py     cards, hand strength, exact and Monte Carlo equity
  game/state.py                          rules engine; the one source of legal actions
  game/tree.py                           betting tree for the solver
  ranges.py                              both players' ranges and their updates
  policy/                                heuristic advisor, push-or-fold, multiway analysis
  solver/vector_cfr.py                   CFR, CFR+, Discounted CFR and exact best response
  solver/toy_games.py                    Kuhn and Leduc poker
  solver/terminal.py, solver/river.py    river payoffs and the river solver
  advisor.py, cli.py                     the entry point and the command line
  live.py, web/                          following a hand as it is played; the local web page
  bots/, arena/                          bots, duplicate matches, Slumbot client
scripts/                                 evaluation, convergence, model error, cross-check, Slumbot analysis
docs/                                    design, results, raw data
tests/                                   143 checks
```

## Limits

- Before the flop (beyond push-or-fold), on the flop and on the turn the bot is heuristic. It is not presented as solved.
- The river solve is only as good as the ranges it is given. A range floor and a limit on all-in size guard against the worst consequences, but they do not make the ranges right.
- The bot does not adapt to its opponent.
- Small edges were not measurable in the time available. See [docs/RESULTS.md](docs/RESULTS.md) for what is and is not shown.
