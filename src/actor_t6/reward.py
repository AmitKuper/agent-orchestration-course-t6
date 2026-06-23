"""Reward shaping for the Q-table actor.

Maps ActionResult outcomes to scalar rewards for each role.
Terminal rewards dominate; step rewards nudge the cop to close in fast
and the thief to stall as long as possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.state import ActionResult

from game.constants import (
    COP,
    THIEF,
    WIN_CAPTURE,
    WIN_COP_TRAPPED,
    WIN_THIEF_SURVIVED,
    WIN_THIEF_TRAPPED,
)

# Terminal rewards per (win_reason, role).
_TERMINAL: dict[str, dict[str, float]] = {
    WIN_CAPTURE:       {COP: 20.0,  THIEF: -20.0},
    WIN_THIEF_TRAPPED: {COP: 20.0,  THIEF: -20.0},
    WIN_THIEF_SURVIVED:{COP: -10.0, THIEF: 10.0},
    WIN_COP_TRAPPED:   {COP: -10.0, THIEF: 10.0},
}

# Per-step rewards (applied on every non-terminal successful action).
_STEP: dict[str, float] = {COP: -0.1, THIEF: 0.1}

# Penalty for illegal / failed actions.
_FAIL_PENALTY: float = -5.0


def compute_reward(result: ActionResult, role: str) -> float:
    """Return scalar reward for the given role based on ActionResult.

    Args:
        result: The ActionResult returned by Game.submit_action().
        role: "cop" or "thief".

    Returns:
        Scalar float reward.

    Raises:
        ValueError: If role is not "cop" or "thief".
    """
    if role not in (COP, THIEF):
        raise ValueError(f"Unknown role: {role!r}. Must be 'cop' or 'thief'.")

    if not result.success:
        return _FAIL_PENALTY

    if result.game_over and result.win_reason in _TERMINAL:
        return _TERMINAL[result.win_reason][role]

    return _STEP[role]
