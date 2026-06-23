"""Self-play training script for CopActor and ThiefActor.

Runs cop vs thief using the alternating freeze schedule from actor_plan.md §12:
  Phase A (0–5k):   CopActor trains vs frozen random ThiefActor
  Phase B (5k–15k): ThiefActor trains vs frozen CopActor
  Phase C (15k–end): both train simultaneously

Usage:
    uv run scripts/train.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Ensure src/ and hw6-common/src/ are on the path when run directly.
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "hw6-common" / "src"))

from game.game import Game  # noqa: E402

from actor_t6.qtable_actor import QTableActor  # noqa: E402

_CONFIG = _ROOT / "config" / "rl_config.json"


def _load_config() -> dict:
    """Load rl_config.json and return as dict."""
    return json.loads(_CONFIG.read_text())


def _sample_positions(
    grid_cols: int, grid_rows: int, rng: random.Random
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Sample non-equal random start positions for cop and thief.

    Args:
        grid_cols: Grid width.
        grid_rows: Grid height.
        rng: Random instance for reproducibility.

    Returns:
        (cop_pos, thief_pos) both as (col, row) tuples.
    """
    cells = [(c, r) for c in range(grid_cols) for r in range(grid_rows)]
    cop_pos = rng.choice(cells)
    thief_pos = rng.choice([p for p in cells if p != cop_pos])
    return cop_pos, thief_pos


def _run_episode(
    game_id: str,
    cop: QTableActor,
    thief: QTableActor,
    cfg: dict,
    rng: random.Random,
    train_cop: bool,
    train_thief: bool,
) -> str | None:
    """Run one full game and apply Q-updates to the actors being trained.

    Args:
        game_id: Unique identifier for this episode.
        cop: CopActor backend.
        thief: ThiefActor backend.
        cfg: Config dict.
        rng: Random instance.
        train_cop: Whether to apply Bellman updates to cop.
        train_thief: Whether to apply Bellman updates to thief.

    Returns:
        Winner string ("cop" or "thief") or None if no winner.
    """
    grid_size = tuple(cfg["grid_size"])
    mechanics = {"max_moves": cfg["max_moves"], "max_barriers": cfg["max_barriers"]}
    cop_pos, thief_pos = _sample_positions(grid_size[0], grid_size[1], rng)
    game = Game.new(game_id, grid_size, cop_pos, thief_pos, mechanics)

    done = False
    while not done:
        for role, backend, do_train in [
            ("cop", cop, train_cop),
            ("thief", thief, train_thief),
        ]:
            obs = game.get_state(role)
            action = backend.get_action(obs)
            result = game.submit_action(role, action)
            if do_train:
                backend.on_result(obs, action, result)
            if result.game_over:
                done = True
                return result.winner
    return None


def _print_progress(episode: int, win_history: list[str | None], cop: QTableActor) -> None:
    """Print a one-line training progress update.

    Args:
        episode: Current episode number.
        win_history: Last N winner strings.
        cop: CopActor (for epsilon display).
    """
    cop_wins = sum(1 for w in win_history if w == "cop")
    rate = cop_wins / len(win_history) if win_history else 0.0
    print(f"  ep {episode:>7,}  cop_win%={rate:.1%}  ε={cop.epsilon:.4f}")


def train() -> None:
    """Run the full self-play training loop and save Q-tables."""
    cfg = _load_config()
    rng = random.Random(42)
    cop = QTableActor(role="cop")
    thief = QTableActor(role="thief")
    thief_random = QTableActor(role="thief")  # frozen random for Phase A

    total = cfg["num_episodes"]
    phase_a_end = 5_000
    phase_b_end = 15_000
    checkpoint = cfg["checkpoint_interval"]
    save_path = Path(cfg["model_save_path"])
    save_path.mkdir(parents=True, exist_ok=True)

    win_history: list[str | None] = []
    log: list[dict] = []

    print(f"Training {total:,} episodes — phases A({phase_a_end:,}) B({phase_b_end:,}) C(rest)")

    for ep in range(1, total + 1):
        train_cop = ep > phase_a_end      # cop learns after warm-up
        train_thief = ep > phase_b_end    # thief learns after cop baseline is set

        # Phase A: cop trains vs frozen random thief
        active_thief = thief if ep > phase_a_end else thief_random

        winner = _run_episode(
            f"ep-{ep}", cop, active_thief, cfg, rng, train_cop, train_thief
        )
        win_history.append(winner)
        if len(win_history) > 1000:
            win_history.pop(0)

        if ep > phase_b_end:
            cop.decay_epsilon()
            thief.decay_epsilon()
        elif ep > phase_a_end:
            cop.decay_epsilon()

        if ep % checkpoint == 0:
            _print_progress(ep, win_history, cop)
            cop.save(save_path / "cop_qtable.npy")
            thief.save(save_path / "thief_qtable.npy")
            cop_win_rate = sum(1 for w in win_history if w == "cop") / len(win_history)
            log.append({"episode": ep, "cop_win_rate": cop_win_rate})

    cop.save(save_path / "cop_qtable.npy")
    thief.save(save_path / "thief_qtable.npy")
    (save_path / "training_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nDone. Models saved to {save_path}/")


if __name__ == "__main__":
    train()
