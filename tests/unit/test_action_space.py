"""Unit tests for actor_t6.action_space."""

import numpy as np
import pytest

from actor_t6.action_space import (
    ACTION_LIST,
    NUM_ACTIONS,
    action_to_idx,
    build_illegal_mask,
    idx_to_action,
)


def test_num_actions() -> None:
    """ACTION_LIST must contain exactly 10 actions (8 dirs + BARRIER + STAY)."""
    assert NUM_ACTIONS == 10


def test_round_trip_all_actions() -> None:
    """Every action survives a str→idx→str round-trip."""
    for action in ACTION_LIST:
        assert idx_to_action(action_to_idx(action)) == action


def test_stay_is_last() -> None:
    """STAY must be the last action (index 9)."""
    assert action_to_idx("STAY") == 9


def test_barrier_index() -> None:
    """BARRIER must be at index 8."""
    assert action_to_idx("BARRIER") == 8


def test_action_to_idx_invalid() -> None:
    """Unknown action string raises KeyError."""
    with pytest.raises(KeyError):
        action_to_idx("INVALID")


def test_idx_to_action_out_of_range() -> None:
    """Out-of-range index raises IndexError."""
    with pytest.raises(IndexError):
        idx_to_action(NUM_ACTIONS)


def test_build_illegal_mask_all_legal() -> None:
    """Mask is all-False when all actions are legal."""
    mask = build_illegal_mask(ACTION_LIST)
    assert not mask.any()


def test_build_illegal_mask_empty() -> None:
    """Mask is all-True when no moves are legal."""
    mask = build_illegal_mask([])
    assert mask.all()


def test_build_illegal_mask_partial() -> None:
    """Only listed legal moves are marked False."""
    legal = ["N", "S"]
    mask = build_illegal_mask(legal)
    assert not mask[action_to_idx("N")]
    assert not mask[action_to_idx("S")]
    assert mask[action_to_idx("NE")]
    assert mask[action_to_idx("BARRIER")]


def test_build_illegal_mask_shape() -> None:
    """Mask shape matches NUM_ACTIONS."""
    mask = build_illegal_mask(["E"])
    assert mask.shape == (NUM_ACTIONS,)
    assert mask.dtype == np.bool_
