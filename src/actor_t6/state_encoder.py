"""State encoder: converts ObservationState to a single integer Q-table index.

Encoding formula:
    state = my_idx + 25 * opp_idx + 25 * 26 * barrier_mask
where:
    my_idx      in [0, 24]  — actor's grid cell (x * rows + y)
    opp_idx     in [0, 25]  — opponent cell or 25 ("not visible")
    barrier_mask in [0, 511] — 9-bit mask of the 3×3 neighbourhood around my_pos
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.state import ObservationState

# Offsets for the 3×3 neighbourhood (centre = my_pos), in row-major order.
_NEIGHBOUR_OFFSETS: list[tuple[int, int]] = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0), (0,  0), (1,  0),
    (-1,  1), (0,  1), (1,  1),
]

NUM_MY_POS: int = 25        # 5×5 grid
NUM_OPP_POS: int = 26       # 25 cells + 25 = "not visible"
NUM_BARRIER_MASKS: int = 512  # 2^9
NUM_STATES: int = NUM_MY_POS * NUM_OPP_POS * NUM_BARRIER_MASKS  # 332,800


def pos_to_idx(pos: tuple[int, int], grid_cols: int) -> int:
    """Convert a grid position to a flat integer index.

    Args:
        pos: (col, row) grid position, 0-indexed.
        grid_cols: Number of columns in the grid.

    Returns:
        Integer index in [0, grid_cols * grid_rows).
    """
    return pos[0] * grid_cols + pos[1]


def opp_to_idx(opp_pos: tuple[int, int] | None, grid_cols: int) -> int:
    """Convert opponent position to an index; returns 25 if not visible.

    Args:
        opp_pos: Opponent (col, row) or None when outside view radius.
        grid_cols: Number of columns in the grid.

    Returns:
        Integer index in [0, 25]; 25 means "not visible".
    """
    if opp_pos is None:
        return NUM_MY_POS  # sentinel for "unknown"
    return pos_to_idx(opp_pos, grid_cols)


def barrier_mask(
    my_pos: tuple[int, int],
    barriers: list[tuple[int, int]],
    grid_cols: int,
    grid_rows: int,
) -> int:
    """Compute a 9-bit integer mask of barriers in the 3×3 neighbourhood.

    Each bit corresponds to one neighbour offset in _NEIGHBOUR_OFFSETS.
    Cells outside the grid boundaries are treated as 0 (no barrier).

    Args:
        my_pos: Actor's current (col, row) position.
        barriers: List of (col, row) barrier positions.
        grid_cols: Number of grid columns.
        grid_rows: Number of grid rows.

    Returns:
        Integer in [0, 511].
    """
    barrier_set = set(barriers)
    mask = 0
    cx, cy = my_pos
    for bit, (dx, dy) in enumerate(_NEIGHBOUR_OFFSETS):
        nx, ny = cx + dx, cy + dy
        if 0 <= nx < grid_cols and 0 <= ny < grid_rows and (nx, ny) in barrier_set:
            mask |= 1 << bit
    return mask


def encode(obs: ObservationState, grid_cols: int = 5, grid_rows: int = 5) -> int:
    """Encode an ObservationState as a single Q-table integer index.

    Args:
        obs: Current observation from the game engine.
        grid_cols: Grid width (default 5).
        grid_rows: Grid height (default 5).

    Returns:
        Integer index in [0, NUM_STATES).
    """
    my_idx = pos_to_idx(obs.my_pos, grid_cols)
    opp_idx = opp_to_idx(obs.opponent_pos, grid_cols)
    bmask = barrier_mask(obs.my_pos, obs.barriers, grid_cols, grid_rows)
    return my_idx + NUM_MY_POS * opp_idx + NUM_MY_POS * NUM_OPP_POS * bmask
