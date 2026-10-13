# pokerbot

A decision bot for heads-up No-Limit Texas Hold'em. Given the cards and the betting situation, it recommends the next move and says how it got there.

```
$ uv run pokerbot advise --hole AsAd --board 9s8s7d6c2h --pot 1000 --to-call 400

Hand AsAd on 9s8s7d6c2h (river, button)
Pot 1000 (10.0 bb), to call 400, stack behind 9700

Recommended: call 400
Strategy for this hand: call 400 100%
Equity against the opponent's estimated range: 32.1%
EV in chips, relative to folding: fold +0, call 400 +49, raise to 1450 -50, raise to 9700 -209
Answered by: river solver layer in 141 ms
Assumptions:
  - heads-up, no rake
  - both players started the hand with 100 big blinds
  - before the flop the button raised to 300 and the big blind called
  - earlier streets were checked through; this is filler and is not used to narrow ranges
  - the big blind bet 400 on this street
Notes:
  - solved 115 iterations and reached the accuracy target
  - exploitability 0.10% of the pot, within this bet tree and these ranges
  - bet tree: 6 decisions, 12 endings
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
- **Useful:** adding the river solver to the heuristic bot wins +39.1 bb/100 (95% interval +26.4 to +51.9) over 20,000 hands.
- **Honest about its limit:** a river solve is accurate to about 0.1% of the pot, but the recommended action changes in 5% to 16% of test situations when the assumed opponent range changes. The assumptions matter more than the arithmetic.
- **Sound:** both bots beat a random bot, an always-call bot and an equity-only bot by wide margins, with no illegal action in 50,000 hands.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest            # 121 checks, about 15 seconds
uv run pytest -m slow    # adds the exhaustive check of all 133,784,560 seven-card hands
```

## Use

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

**A match between two bots**, played as duplicate pairs. Bots: `random`, `call`, `equity`, `heuristic`, `solver`.

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
  bots/, arena/                          bots, duplicate matches, Slumbot client
scripts/                                 evaluation, convergence, model error, cross-check
docs/                                    design, results, raw data
tests/                                   121 checks
```

## Limits

- Before the flop (beyond push-or-fold), on the flop and on the turn the bot is heuristic. It is not presented as solved.
- The river solve is only as good as the ranges it is given.
- The bot does not adapt to its opponent.
- Small edges were not measurable in the time available. See [docs/RESULTS.md](docs/RESULTS.md) for what is and is not shown.
