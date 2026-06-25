"""Action space mapping for the Q-table actor.

Maps action strings to integer indices and back, and builds illegal-action
masks from the legal_moves list provided by ObservationState.
"""

from __future__ import annotations

import numpy as np

# Fixed order: 8 directions, then BARRIER (cop), then STAY (thief).
ACTION_LIST: list[str] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "BARRIER", "STAY"]
NUM_ACTIONS: int = len(ACTION_LIST)

_ACTION_TO_IDX: dict[str, int] = {a: i for i, a in enumerate(ACTION_LIST)}


def action_to_idx(action: str) -> int:
    """Return the integer index for an action string.

    Args:
        action: One of the 9 valid action strings.

    Returns:
        Integer index in [0, NUM_ACTIONS).

    Raises:
        KeyError: If action is not in ACTION_LIST.
    """
    return _ACTION_TO_IDX[action]


def idx_to_action(idx: int) -> str:
    """Return the action string for an integer index.

    Args:
        idx: Integer index in [0, NUM_ACTIONS).

    Returns:
        Action string.

    Raises:
        IndexError: If idx is out of range.
    """
    return ACTION_LIST[idx]


def build_illegal_mask(legal_moves: list[str]) -> np.ndarray:
    """Build a boolean mask where True means the action is illegal.

    Args:
        legal_moves: List of legal action strings from ObservationState.

    Returns:
        Boolean ndarray of shape (NUM_ACTIONS,); True = illegal.
    """
    mask = np.ones(NUM_ACTIONS, dtype=bool)
    for move in legal_moves:
        if move in _ACTION_TO_IDX:
            mask[_ACTION_TO_IDX[move]] = False
    return mask
