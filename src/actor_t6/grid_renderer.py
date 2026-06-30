"""Grid renderer — prints the current game state to stdout for human players."""

from __future__ import annotations

from typing import TYPE_CHECKING

from game.constants import COP, THIEF

if TYPE_CHECKING:
    from game.state import ObservationState

_SYMBOLS = {COP: "C", THIEF: "T"}


def render_observation(obs: ObservationState) -> None:
    """Print the current grid state to stdout.

    Infers grid dimensions from known positions so the display scales
    correctly when view_radius limits what the observer can see.

    Args:
        obs: Current observation; positions determine cell symbols.
    """
    opp = obs.opponent_pos
    cols = max(obs.my_pos[0], *(b[0] for b in obs.barriers), 0,
               *([] if opp is None else [opp[0]])) + 1
    rows = max(obs.my_pos[1], *(b[1] for b in obs.barriers), 0,
               *([] if opp is None else [opp[1]])) + 1
    cols, rows = max(cols, 5), max(rows, 5)

    cop_pos = obs.my_pos if obs.actor == COP else opp
    thief_pos = obs.my_pos if obs.actor == THIEF else opp
    barriers = set(map(tuple, obs.barriers))

    br = (f"  Barriers left: {obs.barriers_remaining}"
          if obs.barriers_remaining is not None else "")
    print(f"\n  Round {obs.round} | You: {obs.actor.upper()} ({_SYMBOLS[obs.actor]})"
          f"  Opponent: {'?' if opp is None else str(opp)}{br}")
    print("    " + " ".join(str(c) for c in range(cols)))
    for r in range(rows):
        row_str = []
        for c in range(cols):
            if cop_pos == (c, r):
                row_str.append("C")
            elif thief_pos == (c, r):
                row_str.append("T")
            elif (c, r) in barriers:
                row_str.append("#")
            else:
                row_str.append(".")
        print(f"  {r} {' '.join(row_str)}")
