# pokerbot

A decision bot for heads-up No-Limit Texas Hold'em. Given the cards and the betting situation, it recommends the next move.

It has three layers:

- a heuristic advisor that answers every spot,
- a CFR solver for river decisions, validated on small games with known solutions,
- experimental extensions that are used only when a measured match shows they help.

The reasoning behind these choices is in [docs/DESIGN.md](docs/DESIGN.md).

## Status

Work in progress. This section is updated as each part lands.

- [x] Design and contracts
- [x] Cards, hand evaluator, equity
- [x] Rules engine, baseline bots, match runner
- [x] Heuristic advisor and command line
- [x] CFR validated on Kuhn and Leduc poker
- [x] River solver
- [x] Push-or-fold game solved
- [x] Slumbot client
- [ ] Results

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest
```

## Use

Ask for advice on one situation. Only the hole cards are required; anything
left out is filled with a stated assumption.

```
uv run pokerbot advise --hole AsKd
uv run pokerbot advise --hole 9h6h --board 7h8h2c --pot 1000 --to-call 400
uv run pokerbot advise --hole AsAd --board 7h8h2c --pot 1000 --to-call 400 --players 4
uv run pokerbot advise --hole QsQd --position bb --pot 350 --to-call 150 --json
```

Chips are in units where the big blind is 100 unless `--bb` says otherwise.
`--pot` includes any bet you are facing.

On the river the answer comes from the solver, with up to `--time` seconds
(default 2). Earlier streets use the heuristic advisor.

Play a duplicate match between two bots (`random`, `call`, `equity`,
`heuristic`, `solver`):

```
uv run pokerbot match heuristic equity --pairs 1000 --workers 16
```

Measure speed on this machine:

```
uv run pokerbot bench
```
