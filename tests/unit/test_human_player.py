"""Unit tests for actor_t6.human_player."""

from unittest.mock import patch

import pytest
from game.state import ActionResult, ObservationState

from actor_t6.human_player import HumanPlayer


def _obs(
    actor: str = "thief",
    my_pos: tuple[int, int] = (2, 2),
    opp_pos: tuple[int, int] | None = (0, 0),
    legal_moves: list[str] | None = None,
) -> ObservationState:
    """Build a minimal ObservationState for testing."""
    return ObservationState(
        actor=actor,
        round=3,
        my_pos=my_pos,
        opponent_pos=opp_pos,
        barriers=[(1, 1)],
        legal_moves=legal_moves or ["N", "S", "E", "W"],
        barriers_remaining=None,
    )


def _result(game_over: bool = False, winner: str | None = None) -> ActionResult:
    """Build a minimal ActionResult for testing."""
    return ActionResult(
        success=True, error=None,
        game_over=game_over, winner=winner, win_reason="capture" if game_over else None,
    )


def test_get_action_valid_move(capsys: pytest.CaptureFixture) -> None:
    """get_action returns the move typed by the user."""
    player = HumanPlayer(role="thief")
    obs = _obs(legal_moves=["N", "S"])
    with patch("builtins.input", return_value="N"):
        action = player.get_action(obs)
    assert action == "N"


def test_get_action_case_insensitive(capsys: pytest.CaptureFixture) -> None:
    """get_action accepts lowercase input."""
    player = HumanPlayer(role="thief")
    obs = _obs(legal_moves=["N", "S"])
    with patch("builtins.input", return_value="n"):
        action = player.get_action(obs)
    assert action == "N"


def test_get_action_reprompts_on_invalid(capsys: pytest.CaptureFixture) -> None:
    """get_action re-prompts on illegal input, then accepts a valid move."""
    player = HumanPlayer(role="thief")
    obs = _obs(legal_moves=["N", "S"])
    responses = iter(["INVALID", "S"])
    with patch("builtins.input", side_effect=responses):
        action = player.get_action(obs)
    assert action == "S"
    captured = capsys.readouterr()
    assert "not a legal move" in captured.out


def test_get_action_quit_raises_system_exit() -> None:
    """Typing 'q' raises SystemExit."""
    player = HumanPlayer(role="thief")
    obs = _obs()
    with patch("builtins.input", return_value="q"), pytest.raises(SystemExit):
        player.get_action(obs)


def test_render_contains_grid_symbols(capsys: pytest.CaptureFixture) -> None:
    """_render output contains player symbol and round number."""
    player = HumanPlayer(role="thief")
    obs = _obs(actor="thief", my_pos=(2, 2), opp_pos=(0, 0))
    player._render(obs)
    captured = capsys.readouterr()
    assert "T" in captured.out
    assert "C" in captured.out
    assert "Round 3" in captured.out


def test_on_result_game_over_prints_outcome(capsys: pytest.CaptureFixture) -> None:
    """on_result prints game-over message when game ends."""
    player = HumanPlayer(role="thief")
    obs = _obs()
    player.on_result(obs, "N", _result(game_over=True, winner="cop"))
    captured = capsys.readouterr()
    assert "Game over" in captured.out
    assert "Actor wins" in captured.out


def test_on_result_no_output_mid_game(capsys: pytest.CaptureFixture) -> None:
    """on_result prints nothing during a non-terminal step."""
    player = HumanPlayer(role="thief")
    obs = _obs()
    player.on_result(obs, "N", _result(game_over=False))
    captured = capsys.readouterr()
    assert captured.out == ""
