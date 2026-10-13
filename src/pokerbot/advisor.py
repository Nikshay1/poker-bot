"""The single entry point: a situation in, a recommendation out.

``advise`` takes a complete, valid ``Spot``. ``spot_from_partial`` builds one
from partial input (for example just the cards and the pot) by filling the
gaps with a named, plausible scenario, and records every assumption made.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from pokerbot.cards import cards_to_str, combo_index
from pokerbot.equity import range_equity
from pokerbot.game.state import (
    CALL,
    CHECK,
    FOLD,
    RAISE,
    STREET_NAMES,
    Action,
    GameConfig,
    HandState,
    Spot,
    SpotError,
)
from pokerbot.game.tree import BetSizing
from pokerbot.policy.base import B_CALL, B_FOLD, B_RAISE, PolicyConfig, Strategy
from pokerbot.ranges import RangeTracker
from pokerbot.solver.river import RiverSolution, solve_river

LAYER_HEURISTIC = "heuristic"
LAYER_RIVER = "river solver"


@dataclass(frozen=True)
class SolverConfig:
    """How the river is solved. ``time_budget`` is in seconds per decision."""

    sizing: BetSizing = field(default_factory=BetSizing)
    variant: str = "dcfr"
    time_budget: float = 0.5
    max_iterations: int = 2000
    target_pct_pot: float | None = 0.25


@dataclass
class Recommendation:
    action: Action
    probabilities: dict[str, float]  # the hand's mixed strategy over its legal actions
    ev: dict[str, float | None]  # chips, relative to folding now; None when not computed
    equity: float | None  # against the opponent's estimated range
    layer: str
    assumptions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    iterations: int | None = None
    seconds: float = 0.0
    raise_to: int | None = None  # heuristic layer: the size behind the "raise" entry

    def to_dict(self) -> dict:
        return {
            "action": self.action.kind,
            "amount": self.action.amount,
            "probabilities": self.probabilities,
            "ev": self.ev,
            "equity": self.equity,
            "layer": self.layer,
            "assumptions": self.assumptions,
            "notes": self.notes,
            "iterations": self.iterations,
            "seconds": round(self.seconds, 4),
            "raise_to": self.raise_to,
        }

    def render(self, spot: Spot) -> str:
        bb = spot.big_blind
        where = f" on {cards_to_str(spot.board)}" if spot.board else ""
        seat = "button" if spot.seat == 0 else "big blind"
        chips = (
            f"Pot {spot.pot} ({spot.pot / bb:.1f} bb), to call {spot.to_call}, "
            f"stack behind {spot.stacks[spot.seat]}"
        )
        lines = [
            f"Hand {cards_to_str(spot.hole)}{where} ({STREET_NAMES[spot.street]}, {seat})",
            chips,
            "",
            f"Recommended: {self.action}",
        ]
        mix = ", ".join(f"{name} {p:.0%}" for name, p in self.probabilities.items() if p > 0.005)
        lines.append(f"Strategy for this hand: {mix}")
        if self.equity is not None:
            lines.append(f"Equity against the opponent's estimated range: {self.equity:.1%}")
        evs = [f"{name} {value:+.0f}" for name, value in self.ev.items() if value is not None]
        if len(evs) > 1:
            lines.append("EV in chips, relative to folding: " + ", ".join(evs))
        lines.append(f"Answered by: {self.layer} layer in {self.seconds * 1000:.0f} ms")
        for title, items in (("Assumptions", self.assumptions), ("Notes", self.notes)):
            if items:
                lines.append(f"{title}:")
                lines.extend(f"  - {item}" for item in items)
        return "\n".join(lines)


def _action_names(spot: Spot, strategy: Strategy) -> dict[int, str]:
    legal = spot.legal()
    names = {}
    if legal.can_fold:
        names[B_FOLD] = FOLD
    names[B_CALL] = CHECK if legal.can_check else CALL
    if legal.can_raise and strategy.raise_to is not None:
        names[B_RAISE] = RAISE
    return names


def recommend_from_strategy(
    spot: Spot,
    strategy: Strategy,
    layer: str,
    rng: np.random.Generator | None = None,
    started: float | None = None,
) -> Recommendation:
    """Read the hero's row out of a whole-range strategy."""
    legal = spot.legal()
    row = strategy.probs[combo_index(*spot.hole)].astype(np.float64).copy()
    names = _action_names(spot, strategy)
    for bucket in range(3):
        if bucket not in names:
            # Probability on an unavailable action moves to check/call.
            row[B_CALL] += row[bucket]
            row[bucket] = 0.0
    row = row / row.sum() if row.sum() > 0 else np.array([0.0, 1.0, 0.0])
    bucket = int(rng.choice(3, p=row)) if rng is not None else int(np.argmax(row))
    action = strategy.action_for(bucket, legal)

    equity = None
    if strategy.equity is not None:
        value = float(strategy.equity[combo_index(*spot.hole)])
        equity = None if np.isnan(value) else value
    ev: dict[str, float | None] = {}
    notes = list(strategy.notes)
    if legal.can_fold:
        ev[FOLD] = 0.0
    if legal.call_amount and equity is not None:
        ev[CALL] = equity * (spot.pot + legal.call_amount) - legal.call_amount
        if spot.street < 3 and legal.call_amount < spot.stacks[spot.seat]:
            notes.append("the call EV assumes no further betting")

    return Recommendation(
        action=action,
        probabilities={names[b]: float(row[b]) for b in sorted(names)},
        ev=ev,
        equity=equity,
        layer=layer,
        assumptions=list(spot.assumptions),
        notes=notes,
        seconds=0.0 if started is None else time.perf_counter() - started,
        raise_to=strategy.action_for(B_RAISE, legal).amount if B_RAISE in names else None,
    )


def recommend_from_solution(
    spot: Spot,
    solution: RiverSolution,
    opp_range: np.ndarray,
    rng: np.random.Generator | None = None,
    started: float | None = None,
) -> Recommendation:
    """Read the hero's hand out of a river solution."""
    k = combo_index(*spot.hole)
    row = solution.strategy[k].astype(np.float64)
    row = row / row.sum()
    choice = int(rng.choice(len(row), p=row)) if rng is not None else int(np.argmax(row))
    action = solution.actions[choice]
    if not spot.legal().allows(action):
        raise RuntimeError(f"the solver produced an illegal action: {action}")
    _, values = solution.for_hand(spot.hole)
    names = [str(a) for a in solution.actions]
    equity = float(range_equity(spot.board, opp_range)[k])
    how = {"converged": "reached the accuracy target", "time": "used its time budget"}.get(
        solution.stopped, "reached the iteration limit"
    )
    accuracy = (
        f"exploitability {solution.exploitability_pct_pot:.2f}% of the pot, "
        "within this bet tree and these ranges"
    )
    notes = [
        f"solved {solution.iterations} iterations and {how}",
        accuracy,
        f"bet tree: {solution.n_decisions} decisions, {solution.n_terminals} endings",
    ]
    return Recommendation(
        action=action,
        probabilities={name: float(p) for name, p in zip(names, row)},
        ev={name: float(v) for name, v in zip(names, values)},
        equity=None if np.isnan(equity) else equity,
        layer=LAYER_RIVER,
        assumptions=list(spot.assumptions),
        notes=notes,
        iterations=solution.iterations,
        seconds=0.0 if started is None else time.perf_counter() - started,
    )


def advise(
    spot: Spot,
    cfg: PolicyConfig | None = None,
    tracker: RangeTracker | None = None,
    rng: np.random.Generator | None = None,
    solver: SolverConfig | None = None,
) -> Recommendation:
    """Recommend an action for a complete heads-up situation.

    With ``rng`` the action is sampled from the hand's mixed strategy (what a
    playing bot should do); without it the most likely action is returned.
    A ``tracker`` kept across the decisions of one hand avoids redoing work.
    With ``solver`` set, river decisions are solved; everything else, and the
    range updates on every street, use the heuristic policy.
    """
    started = time.perf_counter()
    spot.validate()
    tracker = tracker or RangeTracker(cfg)
    tracker.catch_up(spot)
    if solver is not None and spot.street == 3:
        beliefs = tracker.beliefs()
        solution = solve_river(
            spot,
            beliefs,
            sizing=solver.sizing,
            variant=solver.variant,
            time_budget=solver.time_budget,
            max_iterations=solver.max_iterations,
            target_pct_pot=solver.target_pct_pot,
        )
        return recommend_from_solution(spot, solution, beliefs[1 - spot.seat], rng, started)
    strategy = tracker.strategy(spot)
    return recommend_from_strategy(spot, strategy, LAYER_HEURISTIC, rng, started)


def spot_from_partial(
    hole,
    board=(),
    pot: int | None = None,
    to_call: int = 0,
    stack: int | None = None,
    position: str = "button",
    small_blind: int = 50,
    big_blind: int = 100,
    starting_stack_bb: float = 100.0,
) -> Spot:
    """Build a complete situation from partial input.

    The betting that is not given is filled in with the simplest line that
    produces the stated pot: the button raised, the big blind called, earlier
    streets were checked, and any amount to call is a bet made on the current
    street. Each such choice is listed in ``Spot.assumptions``.

    The preflop line is used to set the starting ranges, since the pot size
    supports it. The checks on earlier streets are pure filler and are marked
    in ``Spot.unobserved`` so that ranges are not narrowed by them.
    """
    hole, board = tuple(hole), tuple(board)
    if position not in ("button", "bb"):
        raise SpotError("bad_position", "position must be 'button' or 'bb'")
    if len(board) not in (0, 3, 4, 5):
        raise SpotError("bad_board", "the board must have 0, 3, 4 or 5 cards")
    seat = 0 if position == "button" else 1
    assumptions = ["heads-up, no rake"]
    unobserved: list[int] = []
    start = int(round(starting_stack_bb * big_blind))
    if stack is None:
        assumptions.append(f"both players started the hand with {starting_stack_bb:g} big blinds")

    if not board:
        # Before the flop the blinds fix the state, so only a raise needs describing.
        history: list[tuple[int, int, str, int]] = []
        if seat == 1 and to_call == 0:
            history.append((0, 0, CALL, big_blind - small_blind))
            assumptions.append("the button limped")
        elif seat == 1:
            history.append((0, 0, RAISE, big_blind + to_call))
            assumptions.append(f"the button raised to {big_blind + to_call}")
        elif to_call not in (0, big_blind - small_blind):
            raise SpotError(
                "unsupported_partial",
                "for a button decision after a re-raise, give the full betting history",
            )
        if pot is not None and seat == 1 and pot != 2 * big_blind + to_call:
            raise SpotError("bad_pot", "the pot does not match the blinds and the amount to call")
    else:
        if pot is None:
            raise SpotError("missing_pot", "after the flop the pot size is required")
        if to_call < 0 or to_call >= pot:
            raise SpotError("bad_to_call", "the amount to call must be less than the pot")
        each = (pot - to_call) / 2
        if each != int(each) or each < big_blind:
            raise SpotError(
                "bad_pot",
                "the pot before the current bet must be an even amount of at least two big blinds",
            )
        each = int(each)
        if each == big_blind:
            history = [(0, 0, CALL, big_blind - small_blind), (0, 1, CHECK, 0)]
            assumptions.append("before the flop the button limped and the big blind checked")
        elif each < 2 * big_blind:
            raise SpotError(
                "bad_pot", "the pot is too small for a raised pot and too big for a limped one"
            )
        else:
            history = [(0, 0, RAISE, each), (0, 1, CALL, each - big_blind)]
            assumptions.append(
                f"before the flop the button raised to {each} and the big blind called"
            )
        street = (0, 3, 4, 5).index(len(board))
        for earlier in range(1, street):
            # Filler: nobody saw these checks, so they say nothing about ranges.
            unobserved += [len(history), len(history) + 1]
            history += [(earlier, 1, CHECK, 0), (earlier, 0, CHECK, 0)]
        if street > 1:
            assumptions.append(
                "earlier streets were checked through; this is filler and is not used to narrow ranges"
            )
        if seat == 0:
            if to_call:
                history.append((street, 1, RAISE, to_call))
                assumptions.append(f"the big blind bet {to_call} on this street")
            else:
                history.append((street, 1, CHECK, 0))
                assumptions.append("the big blind checked to the button")
        elif to_call:
            history += [(street, 1, CHECK, 0), (street, 0, RAISE, to_call)]
            assumptions.append(f"the hero checked and the button bet {to_call}")

    # Play the assumed line through the rules engine to get a consistent state.
    if stack is not None:
        # Starting stacks are taken as equal, sized so the hero has ``stack`` left.
        probe = _run_line(hole, board, history, small_blind, big_blind, 10**9)
        start = stack + probe.total_contrib[seat]
        assumptions.append("both players started the hand with the same stack")
    state = _run_line(hole, board, history, small_blind, big_blind, start)
    if state.finished or state.to_act != seat:
        raise SpotError("bad_partial", "the assumed betting line does not leave the hero to act")
    spot = Spot(
        hole=hole,
        board=board,
        seat=seat,
        stacks=tuple(state.stacks),
        street_contrib=tuple(state.street_contrib),
        total_contrib=tuple(state.total_contrib),
        small_blind=small_blind,
        big_blind=big_blind,
        last_raise=state.last_raise,
        history=tuple(state.history),
        assumptions=tuple(assumptions),
        unobserved=tuple(unobserved),
    )
    spot.validate()
    return spot


def _run_line(hole, board, history, small_blind, big_blind, start) -> HandState:
    """Run an assumed betting line through the engine. Hole cards are
    placeholders here; only the public chip counts are read from the result."""
    known = set(hole) | set(board)
    if len(known) != len(hole) + len(board):
        raise SpotError("duplicate_card", "the same card appears twice")
    filler = [c for c in range(52) if c not in known]
    deck = tuple(filler[:4]) + tuple(board) + tuple(filler[4 : 4 + 5 - len(board)])
    try:
        state = HandState(GameConfig(small_blind, big_blind, (start, start)), deck)
        for _, seat, kind, amount in history:
            if state.to_act != seat:
                raise ValueError("the assumed betting line is out of order")
            state.apply(Action(kind, amount))
    except ValueError as err:
        raise SpotError("bad_partial", f"the stated pot and bets are not reachable: {err}") from err
    return state
