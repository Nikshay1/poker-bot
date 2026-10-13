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
- [ ] Cards, hand evaluator, equity
- [ ] Rules engine, baseline bots, match runner
- [ ] Heuristic advisor and command line
- [ ] CFR validated on Kuhn and Leduc poker
- [ ] River solver
- [ ] Results

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest
```
