"""QTableActor — tabular Q-learning backend for Cop & Thief.

Delayed Bellman update: on_result() stores (s, a, r, d) as pending;
get_action() applies it when s' is available from the next observation.
Distance shaping (thief only): shaped_r = r + coeff*(γ*d_after - d_before),
where d = Chebyshev distance to opponent. Dense signal guides the thief
to run away even in unvisited states.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from actor.base_actor import BaseActor

from actor_t6.action_space import NUM_ACTIONS, build_illegal_mask, idx_to_action
from actor_t6.reward import compute_reward
from actor_t6.state_encoder import NUM_STATES, encode

if TYPE_CHECKING:
    from game.state import ActionResult, ObservationState

_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config" / "rl_config.json"
_THIEF = "thief"


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Return the Chebyshev (L∞) distance between two grid positions."""
    return float(max(abs(a[0] - b[0]), abs(a[1] - b[1])))


class QTableActor(BaseActor):
    """Q-table actor for a fixed role (cop or thief).

    Uses delayed Bellman updates and optional distance-based reward shaping
    for the thief to encourage evasion even in sparsely visited states.
    """

    def __init__(self, role: str, config_path: Path | str = _DEFAULT_CONFIG) -> None:
        """Initialise Q-table and hyperparameters from config.

        Args:
            role: "cop" or "thief".
            config_path: Path to rl_config.json.
        """
        self.role = role
        cfg = json.loads(Path(config_path).read_text())
        self._alpha: float = cfg["learning_rate"]
        self._gamma: float = cfg["discount_factor"]
        self.epsilon: float = cfg["epsilon_start"]
        self._eps_decay: float = cfg["epsilon_decay"]
        self._eps_min: float = cfg["epsilon_min"]
        self._grid_cols: int = cfg["grid_size"][0]
        self._grid_rows: int = cfg["grid_size"][1]
        # Distance shaping coefficient: thief is rewarded for increasing distance from cop.
        self._dist_coeff: float = cfg.get("distance_shaping_coeff", 2.0) if role == _THIEF else 0.0
        self._q: np.ndarray = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float32)
        self._last_state: int = 0
        self._last_action: int = 0
        # Pending deferred update: (state, action_idx, reward, dist_before) or None.
        self._pending: tuple[int, int, float, float] | None = None

    def get_action(self, obs: ObservationState) -> str:
        """Select action via ε-greedy; apply any deferred Bellman update first.

        Args:
            obs: Current observation from the game engine.

        Returns:
            A legal action string from obs.legal_moves.
        """
        current_state = encode(obs, self._grid_cols, self._grid_rows)

        if self._pending is not None:
            s, a, r, d_before = self._pending
            if self._dist_coeff and obs.opponent_pos is not None:
                d_after = _chebyshev(obs.my_pos, obs.opponent_pos)
                r = r + self._dist_coeff * (self._gamma * d_after - d_before)
            best_next = float(np.max(self._q[current_state]))
            self._q[s, a] += self._alpha * (r + self._gamma * best_next - self._q[s, a])
            self._pending = None

        illegal = build_illegal_mask(obs.legal_moves)
        if random.random() < self.epsilon:
            action_idx = random.choice([i for i in range(NUM_ACTIONS) if not illegal[i]])
        else:
            q_row = self._q[current_state].copy().astype(np.float64)
            q_row[illegal] = -np.inf
            # Tiny noise breaks ties among equal Q-values (avoids always picking action 0).
            q_row += np.random.uniform(-1e-8, 1e-8, NUM_ACTIONS)
            action_idx = int(np.argmax(q_row))

        self._last_state = current_state
        self._last_action = action_idx
        return idx_to_action(action_idx)

    def on_result(self, obs: ObservationState, action: str, result: ActionResult) -> None:
        """Defer Bellman update (non-terminal) or apply immediately (terminal).

        Args:
            obs: Pre-action observation (state s, not s').
            action: Action string submitted to the game engine.
            result: ActionResult returned by the game engine.
        """
        reward = compute_reward(result, self.role)
        s, a = self._last_state, self._last_action
        if result.game_over:
            self._q[s, a] += self._alpha * (reward - self._q[s, a])
            self._pending = None
        else:
            d = _chebyshev(obs.my_pos, obs.opponent_pos) if obs.opponent_pos is not None else 0.0
            self._pending = (s, a, reward, d)

    def reset_episode(self) -> None:
        """Discard pending update at episode boundaries to avoid cross-episode pollution."""
        self._pending = None

    def decay_epsilon(self) -> None:
        """Decay exploration rate by one step, flooring at epsilon_min."""
        self.epsilon = max(self._eps_min, self.epsilon * self._eps_decay)

    def save(self, path: str | Path) -> None:
        """Save Q-table to a .npy file."""
        np.save(str(path), self._q)

    @classmethod
    def load(
        cls, role: str, path: str | Path, config_path: Path | str = _DEFAULT_CONFIG
    ) -> QTableActor:
        """Load a trained Q-table; sets epsilon=0 for pure exploitation.

        Args:
            role: "cop" or "thief".
            path: Path to the saved .npy file.
            config_path: Path to rl_config.json.

        Returns:
            QTableActor with loaded Q-table and epsilon=0.
        """
        actor = cls(role=role, config_path=config_path)
        actor._q = np.load(str(path))
        actor.epsilon = 0.0
        return actor
