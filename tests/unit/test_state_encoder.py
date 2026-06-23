"""Unit tests for actor_t6.state_encoder."""


from game.state import ObservationState

from actor_t6.state_encoder import (
    NUM_STATES,
    barrier_mask,
    encode,
    opp_to_idx,
    pos_to_idx,
)


def _obs(
    my_pos: tuple[int, int],
    opp_pos: tuple[int, int] | None = None,
    barriers: list[tuple[int, int]] | None = None,
) -> ObservationState:
    """Build a minimal ObservationState for testing."""
    return ObservationState(
        actor="cop",
        round=0,
        my_pos=my_pos,
        opponent_pos=opp_pos,
        barriers=barriers or [],
        legal_moves=["N", "S"],
        barriers_remaining=5,
    )


def test_num_states() -> None:
    """NUM_STATES must equal 25 * 26 * 512."""
    assert NUM_STATES == 25 * 26 * 512


def test_pos_to_idx_origin() -> None:
    """(0,0) maps to index 0."""
    assert pos_to_idx((0, 0), 5) == 0


def test_pos_to_idx_last_cell() -> None:
    """(4,4) maps to index 24 on a 5-col grid."""
    assert pos_to_idx((4, 4), 5) == 24


def test_opp_to_idx_visible() -> None:
    """Visible opponent delegates to pos_to_idx."""
    assert opp_to_idx((2, 3), 5) == pos_to_idx((2, 3), 5)


def test_opp_to_idx_not_visible() -> None:
    """None opponent returns sentinel 25."""
    assert opp_to_idx(None, 5) == 25


def test_barrier_mask_empty() -> None:
    """No barriers yields mask 0."""
    assert barrier_mask((2, 2), [], 5, 5) == 0


def test_barrier_mask_neighbour() -> None:
    """A barrier at a neighbour cell sets the correct bit."""
    # _NEIGHBOUR_OFFSETS[1] = (0, -1) → cell (2, 1)
    m = barrier_mask((2, 2), [(2, 1)], 5, 5)
    assert m & (1 << 1)


def test_barrier_mask_out_of_bounds() -> None:
    """Out-of-bounds neighbour positions do not raise and contribute 0."""
    m = barrier_mask((0, 0), [], 5, 5)
    assert m == 0


def test_encode_in_range() -> None:
    """encode() output is always in [0, NUM_STATES)."""
    obs = _obs((1, 2), (3, 4), [(0, 0)])
    idx = encode(obs)
    assert 0 <= idx < NUM_STATES


def test_encode_different_positions() -> None:
    """Different my_pos values produce different indices."""
    obs1 = _obs((0, 0))
    obs2 = _obs((1, 0))
    assert encode(obs1) != encode(obs2)


def test_encode_opp_visible_vs_hidden() -> None:
    """Same position, visible vs hidden opponent → different indices."""
    obs_vis = _obs((2, 2), (3, 3))
    obs_hid = _obs((2, 2), None)
    assert encode(obs_vis) != encode(obs_hid)


def test_encode_barrier_changes_index() -> None:
    """Adding a barrier changes the encoded state."""
    obs_no = _obs((2, 2))
    obs_bar = _obs((2, 2), barriers=[(2, 1)])
    assert encode(obs_no) != encode(obs_bar)


def test_encode_deterministic() -> None:
    """Same ObservationState always produces the same index."""
    obs = _obs((1, 1), (3, 3), [(0, 0)])
    assert encode(obs) == encode(obs)
