"""QTableActor — tabular Q-learning backend for Cop & Thief.

Implements BaseActor using an ε-greedy policy over a numpy Q-table.
Bellman updates happen in on_result(); get_action() caches (s, a) for it.
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


class QTableActor(BaseActor):
    """Q-table actor for a fixed role (cop or thief).

    The Q-table has shape (NUM_STATES, NUM_ACTIONS). Each call to
    get_action() stores (state_idx, action_idx) so on_result() can
    apply the Bellman update without re-encoding state.
    """

    def __init__(self, role: str, config_path: Path | str = _DEFAULT_CONFIG) -> None:
        """Initialise Q-table and hyperparameters from config.

        Args:
            role: "cop" or "thief" — determines which role this actor plays.
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
        self._q: np.ndarray = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float32)
        self._last_state: int = 0
        self._last_action: int = 0

    def get_action(self, obs: ObservationState) -> str:
        """Select an action via ε-greedy policy, caching (s, a) for update.

        Args:
            obs: Current observation from the game engine.

        Returns:
            A legal action string from obs.legal_moves.
        """
        state = encode(obs, self._grid_cols, self._grid_rows)
        illegal = build_illegal_mask(obs.legal_moves)

        if random.random() < self.epsilon:
            action_idx = random.choice(
                [i for i in range(NUM_ACTIONS) if not illegal[i]]
            )
        else:
            q_row = self._q[state].copy().astype(np.float64)
            q_row[illegal] = -np.inf
            action_idx = int(np.argmax(q_row))

        self._last_state = state
        self._last_action = action_idx
        return idx_to_action(action_idx)

    def on_result(
        self,
        obs: ObservationState,
        action: str,
        result: ActionResult,
    ) -> None:
        """Apply Bellman update using the transition from the last get_action() call.

        Args:
            obs: The observation that was passed to get_action().
            action: The action string that was submitted.
            result: The ActionResult returned by the game engine.
        """
        reward = compute_reward(result, self.role)
        if result.game_over:
            best_next = 0.0
        else:
            next_state = encode(obs, self._grid_cols, self._grid_rows)
            best_next = float(np.max(self._q[next_state]))

        td_target = reward + self._gamma * best_next
        td_error = td_target - self._q[self._last_state, self._last_action]
        self._q[self._last_state, self._last_action] += self._alpha * td_error

    def decay_epsilon(self) -> None:
        """Decay exploration rate by one step, flooring at epsilon_min."""
        self.epsilon = max(self._eps_min, self.epsilon * self._eps_decay)

    def save(self, path: str | Path) -> None:
        """Save Q-table to a .npy file.

        Args:
            path: Destination file path.
        """
        np.save(str(path), self._q)

    @classmethod
    def load(
        cls,
        role: str,
        path: str | Path,
        config_path: Path | str = _DEFAULT_CONFIG,
    ) -> QTableActor:
        """Load a trained Q-table from a .npy file.

        Args:
            role: "cop" or "thief".
            path: Path to the saved .npy file.
            config_path: Path to rl_config.json.

        Returns:
            QTableActor with the loaded Q-table and epsilon set to 0.
        """
        actor = cls(role=role, config_path=config_path)
        actor._q = np.load(str(path))
        actor.epsilon = 0.0
        return actor
