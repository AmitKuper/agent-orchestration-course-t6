"""Self-play training script for CopActor and ThiefActor.

Phase schedule (thief-first so it learns escape before facing expert cop):
  Phase A (0–10k):   Thief trains vs frozen random CopActor
  Phase B (10k–20k): Cop trains vs frozen (partially trained) Thief
  Phase C (20k–end): Both train simultaneously

Usage:
    uv run scripts/train.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

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
    active_cop: QTableActor,
    active_thief: QTableActor,
    cfg: dict,
    rng: random.Random,
    train_cop: bool,
    train_thief: bool,
) -> str | None:
    """Run one full game and apply Q-updates to the actors being trained.

    Args:
        game_id: Unique identifier for this episode.
        active_cop: CopActor backend (may be random proxy).
        active_thief: ThiefActor backend (may be random proxy).
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
    active_cop.reset_episode()
    active_thief.reset_episode()
    game = Game.new(game_id, grid_size, cop_pos, thief_pos, mechanics)

    done = False
    while not done:
        for role, backend, do_train in [
            ("thief", active_thief, train_thief),
            ("cop", active_cop, train_cop),
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


def _print_progress(episode: int, win_history: list[str | None], cop: QTableActor,
                    thief: QTableActor) -> None:
    """Print a one-line training progress update.

    Args:
        episode: Current episode number.
        win_history: Last N winner strings.
        cop: CopActor (for epsilon display).
        thief: ThiefActor (for epsilon display).
    """
    cop_wins = sum(1 for w in win_history if w == "cop")
    rate = cop_wins / len(win_history) if win_history else 0.0
    print(f"  ep {episode:>7,}  cop_win%={rate:.1%}  "
          f"cop_eps={cop.epsilon:.3f}  thief_eps={thief.epsilon:.3f}")


def train() -> None:
    """Run the full self-play training loop and save Q-tables."""
    cfg = _load_config()
    rng = random.Random(42)
    cop = QTableActor(role="cop")
    thief = QTableActor(role="thief")
    cop_random = QTableActor(role="cop")    # stays at epsilon=1 (never decays)

    total = cfg["num_episodes"]
    phase_a_end = 10_000   # thief warms up vs random cop
    phase_b_end = 20_000   # cop warms up vs frozen thief
    checkpoint = cfg["checkpoint_interval"]
    save_path = Path(cfg["model_save_path"])
    save_path.mkdir(parents=True, exist_ok=True)

    win_history: list[str | None] = []
    log: list[dict] = []
    print(f"Training {total:,} episodes — A(thief,{phase_a_end:,}) B(cop,{phase_b_end:,}) C(both)")

    for ep in range(1, total + 1):
        if ep <= phase_a_end:
            active_cop, active_thief = cop_random, thief
            train_cop, train_thief = False, True
        elif ep <= phase_b_end:
            active_cop, active_thief = cop, thief
            train_cop, train_thief = True, False
        else:
            active_cop, active_thief = cop, thief
            train_cop, train_thief = True, True

        winner = _run_episode(
            f"ep-{ep}", active_cop, active_thief, cfg, rng, train_cop, train_thief
        )
        win_history.append(winner)
        if len(win_history) > 1000:
            win_history.pop(0)

        if ep <= phase_a_end:
            thief.decay_epsilon()
        elif ep <= phase_b_end:
            cop.decay_epsilon()
        else:
            cop.decay_epsilon()
            thief.decay_epsilon()

        if ep % checkpoint == 0:
            _print_progress(ep, win_history, cop, thief)
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
