"""Unit tests for actor_t6.reward."""

import pytest
from game.state import ActionResult

from actor_t6.reward import compute_reward


def _result(
    success: bool = True,
    game_over: bool = False,
    winner: str | None = None,
    win_reason: str | None = None,
) -> ActionResult:
    """Build a minimal ActionResult for testing."""
    return ActionResult(
        success=success,
        error=None if success else "err",
        game_over=game_over,
        winner=winner,
        win_reason=win_reason,
    )


# --- Terminal outcomes ---

def test_cop_wins_capture() -> None:
    """Cop gets +20 on capture."""
    r = _result(game_over=True, winner="cop", win_reason="capture")
    assert compute_reward(r, "cop") == 20.0


def test_thief_loses_capture() -> None:
    """Thief gets -20 on capture."""
    r = _result(game_over=True, winner="cop", win_reason="capture")
    assert compute_reward(r, "thief") == -20.0


def test_cop_wins_thief_trapped() -> None:
    """Cop gets +20 when thief is trapped."""
    r = _result(game_over=True, winner="cop", win_reason="thief_trapped")
    assert compute_reward(r, "cop") == 20.0


def test_thief_survives() -> None:
    """Thief gets +10 for surviving; cop gets -10."""
    r = _result(game_over=True, winner="thief", win_reason="thief_survived")
    assert compute_reward(r, "thief") == 10.0
    assert compute_reward(r, "cop") == -10.0


def test_cop_trapped() -> None:
    """Thief gets +10 when cop is trapped; cop gets -10."""
    r = _result(game_over=True, winner="thief", win_reason="cop_trapped")
    assert compute_reward(r, "thief") == 10.0
    assert compute_reward(r, "cop") == -10.0


# --- Step reward ---

def test_step_reward_cop() -> None:
    """Cop receives -0.5 per non-terminal step (pressure to capture quickly)."""
    r = _result()
    assert compute_reward(r, "cop") == pytest.approx(-0.5)


def test_step_reward_thief() -> None:
    """Thief receives +1.0 per non-terminal step (survival signal on scale with -20 capture)."""
    r = _result()
    assert compute_reward(r, "thief") == pytest.approx(1.0)


# --- Failure penalty ---

def test_failed_action_cop() -> None:
    """Failed action yields -5 for cop."""
    r = _result(success=False)
    assert compute_reward(r, "cop") == -5.0


def test_failed_action_thief() -> None:
    """Failed action yields -5 for thief."""
    r = _result(success=False)
    assert compute_reward(r, "thief") == -5.0


# --- Invalid role ---

def test_invalid_role_raises() -> None:
    """Unknown role raises ValueError."""
    r = _result()
    with pytest.raises(ValueError, match="Unknown role"):
        compute_reward(r, "referee")
