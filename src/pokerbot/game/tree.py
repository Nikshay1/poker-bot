"""The betting tree of one street, built from a list of bet sizes.

No-Limit Hold'em allows any bet size, which makes the true tree far too big
to solve. The tree here keeps a few sizes per decision. It starts from the
actual situation, so a bet the opponent has already made is in it exactly;
only the choices still to come are restricted.

Legality comes from ``legal_actions``, the same function the engine uses.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from pokerbot.game.state import CALL, CHECK, FOLD, RAISE, Action, Legal, Spot, legal_actions


@dataclass(frozen=True)
class BetSizing:
    bet_fracs: tuple[float, ...] = (0.33, 0.75)  # opening bets, as fractions of the pot
    raise_fracs: tuple[float, ...] = (0.75,)  # raises, as fractions of the pot after calling
    all_in: bool = True  # always offer all-in when a raise is legal
    max_raises: int = 3  # bets and raises allowed below the root
    commit_frac: float = 0.67  # a size using this share of the stack becomes all-in


@dataclass(frozen=True)
class BetState:
    """The public chip state within one street."""

    street_contrib: tuple[int, int]
    total_contrib: tuple[int, int]
    stacks: tuple[int, int]
    to_act: int
    last_raise: int
    big_blind: int
    acted: tuple[bool, bool] = (False, False)
    raises: int = 0  # bets and raises made below the root
    folded: int | None = None
    closed: bool = False  # betting is over without a fold

    @staticmethod
    def from_spot(spot: Spot) -> BetState:
        # Anyone who has put chips in this street, or checked, has acted. The
        # hero is about to act, so only the opponent's flag matters.
        opp = 1 - spot.seat
        opp_acted = any(s == spot.street and seat == opp for s, seat, _, _ in spot.history)
        acted = [False, False]
        acted[opp] = opp_acted
        return BetState(
            street_contrib=spot.street_contrib,
            total_contrib=spot.total_contrib,
            stacks=spot.stacks,
            to_act=spot.seat,
            last_raise=spot.last_raise,
            big_blind=spot.big_blind,
            acted=tuple(acted),
        )

    @property
    def pot(self) -> int:
        return self.total_contrib[0] + self.total_contrib[1]

    @property
    def finished(self) -> bool:
        return self.closed or self.folded is not None

    def legal(self) -> Legal:
        return legal_actions(
            self.street_contrib, self.stacks, self.to_act, self.last_raise, self.big_blind
        )

    def apply(self, action: Action) -> BetState:
        seat, opp = self.to_act, 1 - self.to_act
        legal = self.legal()
        if not legal.allows(action):
            raise ValueError(f"illegal action {action}")
        if action.kind == FOLD:
            return replace(self, folded=seat)
        street, total, stacks = (
            list(self.street_contrib),
            list(self.total_contrib),
            list(self.stacks),
        )
        acted = list(self.acted)
        acted[seat] = True
        last_raise, raises = self.last_raise, self.raises
        if action.kind == CALL:
            added = legal.call_amount
        elif action.kind == RAISE:
            last_raise = max(last_raise, action.amount - street[opp])
            added = action.amount - street[seat]
            raises += 1
        else:
            added = 0
        street[seat] += added
        total[seat] += added
        stacks[seat] -= added
        matched = street[0] == street[1]
        closed = matched and (all(acted) or min(stacks) == 0)
        return replace(
            self,
            street_contrib=tuple(street),
            total_contrib=tuple(total),
            stacks=tuple(stacks),
            to_act=opp,
            last_raise=last_raise,
            acted=tuple(acted),
            raises=raises,
            closed=closed,
        )


def tree_actions(state: BetState, sizing: BetSizing) -> list[Action]:
    """The actions kept in the tree at this state, in a fixed order:
    fold, check or call, then raises from smallest to largest."""
    legal = state.legal()
    me, opp = state.to_act, 1 - state.to_act
    actions = []
    if legal.can_fold:
        actions.append(Action(FOLD))
    actions.append(Action(CHECK) if legal.can_check else Action(CALL, legal.call_amount))
    if not legal.can_raise or state.raises >= sizing.max_raises:
        return actions

    pot, to_call = state.pot, legal.call_amount
    if to_call == 0:
        targets = [state.street_contrib[me] + f * pot for f in sizing.bet_fracs]
    else:
        targets = [state.street_contrib[opp] + f * (pot + to_call) for f in sizing.raise_fracs]
    can_add = legal.max_raise_to - state.street_contrib[me]
    sizes = set()
    for target in targets:
        raise_to = legal.clamp_raise(int(round(target)))
        if raise_to - state.street_contrib[me] >= sizing.commit_frac * can_add:
            raise_to = legal.max_raise_to
        sizes.add(raise_to)
    if sizing.all_in:
        sizes.add(legal.max_raise_to)
    actions.extend(Action(RAISE, size) for size in sorted(sizes))
    return actions
