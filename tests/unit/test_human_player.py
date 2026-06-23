"""Unit tests for actor_t6.human_player."""

from unittest.mock import patch

import pytest
from game.state import ActionResult, ObservationState

from actor_t6.human_player import _CHAR_MAP, _SCAN_MAP, HumanPlayer


def _obs(
    actor: str = "thief",
    my_pos: tuple[int, int] = (2, 2),
    opp_pos: tuple[int, int] | None = (0, 0),
    legal_moves: list[str] | None = None,
) -> ObservationState:
    """Build a minimal ObservationState for testing."""
    return ObservationState(
        actor=actor, round=3, my_pos=my_pos, opponent_pos=opp_pos,
        barriers=[(1, 1)], legal_moves=legal_moves or ["N", "S", "E", "W"],
        barriers_remaining=None,
    )


def _result(game_over: bool = False, winner: str | None = None) -> ActionResult:
    """Build a minimal ActionResult for testing."""
    return ActionResult(
        success=True, error=None,
        game_over=game_over, winner=winner, win_reason="capture" if game_over else None,
    )


# --- Key map sanity checks ---

def test_char_map_covers_all_directions() -> None:
    """_CHAR_MAP must cover 8 directions + BARRIER."""
    assert set(_CHAR_MAP.values()) == {"N", "NE", "E", "SE", "S", "SW", "W", "NW", "BARRIER"}


def test_scan_map_covers_all_directions() -> None:
    """_SCAN_MAP must cover 8 directions + BARRIER."""
    assert set(_SCAN_MAP.values()) == {"N", "NE", "E", "SE", "S", "SW", "W", "NW", "BARRIER"}


# --- NumLock ON: ASCII digit path ---

def test_get_action_numlock_on_digit(capsys: pytest.CaptureFixture) -> None:
    """NumLock ON: pressing '8' (N) is accepted."""
    player = HumanPlayer(role="thief")
    with patch("msvcrt.getch", side_effect=[b"8"]):
        action = player.get_action(_obs(legal_moves=["N", "S"]))
    assert action == "N"


def test_get_action_numlock_on_digit_sw(capsys: pytest.CaptureFixture) -> None:
    """NumLock ON: pressing '1' maps to SW."""
    player = HumanPlayer(role="thief")
    with patch("msvcrt.getch", side_effect=[b"1"]):
        action = player.get_action(_obs(legal_moves=["SW", "S"]))
    assert action == "SW"


# --- NumLock OFF: extended scan-code path ---

def test_get_action_numlock_off_up(capsys: pytest.CaptureFixture) -> None:
    """NumLock OFF: Up-arrow scan (0x48) maps to N."""
    player = HumanPlayer(role="thief")
    # msvcrt yields \xe0 then scan byte 0x48
    with patch("msvcrt.getch", side_effect=[b"\xe0", bytes([0x48])]):
        action = player.get_action(_obs(legal_moves=["N", "S"]))
    assert action == "N"


def test_get_action_numlock_off_diagonal(capsys: pytest.CaptureFixture) -> None:
    """NumLock OFF: PgDn scan (0x51) maps to SE."""
    player = HumanPlayer(role="thief")
    with patch("msvcrt.getch", side_effect=[b"\xe0", bytes([0x51])]):
        action = player.get_action(_obs(legal_moves=["SE", "E"]))
    assert action == "SE"


# --- Illegal / unknown key handling ---

def test_unknown_key_ignored(capsys: pytest.CaptureFixture) -> None:
    """Unrecognised key is ignored; next valid key accepted."""
    player = HumanPlayer(role="thief")
    # 'Z' is unknown, then '2' maps to S
    with patch("msvcrt.getch", side_effect=[b"Z", b"2"]):
        action = player.get_action(_obs(legal_moves=["S"]))
    assert action == "S"


def test_illegal_move_reprompts(capsys: pytest.CaptureFixture) -> None:
    """Valid key whose action is not in legal_moves shows re-prompt message."""
    player = HumanPlayer(role="thief")
    # '3'=SE not legal; then '8'=N is legal
    with patch("msvcrt.getch", side_effect=[b"3", b"8"]):
        action = player.get_action(_obs(legal_moves=["N"]))
    assert action == "N"
    assert "not available" in capsys.readouterr().out


def test_quit_raises_system_exit() -> None:
    """Pressing 'q' raises SystemExit."""
    player = HumanPlayer(role="thief")
    with patch("msvcrt.getch", side_effect=[b"q"]), pytest.raises(SystemExit):
        player.get_action(_obs())


# --- Render and on_result ---

def test_render_contains_symbols(capsys: pytest.CaptureFixture) -> None:
    """_render output shows both player symbols and round number."""
    player = HumanPlayer(role="thief")
    player._render(_obs(actor="thief", my_pos=(2, 2), opp_pos=(0, 0)))
    out = capsys.readouterr().out
    assert "T" in out and "C" in out and "Round 3" in out


def test_on_result_game_over(capsys: pytest.CaptureFixture) -> None:
    """on_result prints game-over message when game ends."""
    player = HumanPlayer(role="thief")
    player.on_result(_obs(), "N", _result(game_over=True, winner="cop"))
    assert "Game over" in capsys.readouterr().out


def test_on_result_mid_game_silent(capsys: pytest.CaptureFixture) -> None:
    """on_result prints nothing during a non-terminal step."""
    player = HumanPlayer(role="thief")
    player.on_result(_obs(), "N", _result(game_over=False))
    assert capsys.readouterr().out == ""
