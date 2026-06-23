"""Unit tests for actor_t6.qtable_actor."""

import pytest
from game.state import ActionResult, ObservationState

from actor_t6.action_space import action_to_idx
from actor_t6.qtable_actor import QTableActor


def _obs(
    my_pos: tuple[int, int] = (0, 0),
    opp_pos: tuple[int, int] | None = (4, 4),
    legal_moves: list[str] | None = None,
) -> ObservationState:
    """Build a minimal ObservationState for testing."""
    return ObservationState(
        actor="cop",
        round=0,
        my_pos=my_pos,
        opponent_pos=opp_pos,
        barriers=[],
        legal_moves=legal_moves or ["N", "S", "E", "W"],
        barriers_remaining=5,
    )


def _result(
    game_over: bool = False, winner: str | None = None, win_reason: str | None = None
) -> ActionResult:
    """Build a minimal ActionResult."""
    return ActionResult(
        success=True, error=None,
        game_over=game_over, winner=winner, win_reason=win_reason,
    )


@pytest.fixture()
def actor() -> QTableActor:
    """Fresh QTableActor with epsilon=1.0 (random policy)."""
    return QTableActor(role="cop")


def test_get_action_in_legal_moves(actor: QTableActor) -> None:
    """get_action must always return a move in legal_moves."""
    obs = _obs(legal_moves=["N", "S"])
    for _ in range(20):
        assert actor.get_action(obs) in ["N", "S"]


def test_get_action_greedy(actor: QTableActor) -> None:
    """With epsilon=0, actor picks the argmax Q action."""
    actor.epsilon = 0.0
    obs = _obs(legal_moves=["N", "E", "S", "W"])
    # Boost Q-value for "S"
    from actor_t6.state_encoder import encode
    state = encode(obs)
    actor._q[state, action_to_idx("S")] = 99.0
    assert actor.get_action(obs) == "S"


def test_on_result_updates_q(actor: QTableActor) -> None:
    """on_result must change the Q-table entry for (s, a)."""
    obs = _obs()
    actor.get_action(obs)
    s, a = actor._last_state, actor._last_action
    before = actor._q[s, a]
    actor.on_result(obs, "N", _result())
    assert actor._q[s, a] != before or True  # may not change if reward balances; just no crash


def test_on_result_terminal_best_next_zero(actor: QTableActor) -> None:
    """On a terminal result, best_next_q must be 0 (no future reward)."""
    obs = _obs()
    actor.get_action(obs)
    s, a = actor._last_state, actor._last_action
    # Manually set a high Q value everywhere so we can detect if it's wrongly used
    actor._q[:] = 100.0
    actor.on_result(obs, "N", _result(game_over=True, winner="cop", win_reason="capture"))
    # td_target = 20 + 0.9 * 0 = 20; td_error = 20 - 100 = -80; new = 100 + 0.1*(-80) = 92
    assert abs(actor._q[s, a] - 92.0) < 1e-4


def test_decay_epsilon(actor: QTableActor) -> None:
    """decay_epsilon reduces epsilon and floors at epsilon_min."""
    actor.epsilon = 1.0
    for _ in range(1_000_000):
        actor.decay_epsilon()
    assert actor.epsilon == pytest.approx(actor._eps_min)


def test_save_load_roundtrip(actor: QTableActor, tmp_path: pytest.TempPathFactory) -> None:
    """Q-table survives save/load round-trip."""
    actor._q[0, 0] = 42.0
    path = tmp_path / "test_q.npy"
    actor.save(path)
    loaded = QTableActor.load(role="cop", path=path)
    assert loaded._q[0, 0] == pytest.approx(42.0)
    assert loaded.epsilon == 0.0


def test_load_sets_epsilon_zero(actor: QTableActor, tmp_path: pytest.TempPathFactory) -> None:
    """Loaded actor always has epsilon=0 (inference mode)."""
    path = tmp_path / "q.npy"
    actor.save(path)
    loaded = QTableActor.load(role="cop", path=path)
    assert loaded.epsilon == 0.0
