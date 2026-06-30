"""Thief-focused trainer: strengthen the thief against the frozen expert cop.

The cop Q-table reached ~99% win in self-play, so the bottleneck is the thief.
This script keeps cop_qtable.npy unchanged, loads it as a frozen (epsilon=0)
opponent, and trains ONLY the thief against that expert cop — then saves the
improved thief_qtable.npy. Self-play helpers are reused from train.py.

Usage:
    uv run scripts/train_thief.py [num_episodes]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "hw6-common" / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from train import _load_config, _print_progress, _run_episode  # noqa: E402

from actor_t6.qtable_actor import QTableActor  # noqa: E402


def train_thief(num_episodes: int | None = None) -> None:
    """Train only the thief against the frozen expert cop; save the thief table.

    Args:
        num_episodes: Episode count; falls back to rl_config.json num_episodes.
    """
    cfg = _load_config()
    rng = random.Random(42)
    save_path = Path(cfg["model_save_path"])
    cop_path = save_path / "cop_qtable.npy"
    if not cop_path.exists():
        raise FileNotFoundError(f"Expert cop table not found: {cop_path}")

    # Frozen expert cop: epsilon=0 (pure exploitation), never updated or decayed.
    cop = QTableActor.load("cop", cop_path)
    thief = QTableActor(role="thief")  # fresh thief, full exploration schedule

    total = num_episodes or cfg["num_episodes"]
    checkpoint = cfg["checkpoint_interval"]
    win_history: list[str | None] = []
    print(f"Training thief {total:,} episodes vs frozen expert cop (eps0)")

    for ep in range(1, total + 1):
        winner = _run_episode(
            f"thief-ep-{ep}", cop, thief, cfg, rng,
            train_cop=False, train_thief=True,
        )
        win_history.append(winner)
        if len(win_history) > 1000:
            win_history.pop(0)
        thief.decay_epsilon()

        if ep % checkpoint == 0:
            _print_progress(ep, win_history, cop, thief)
            thief.save(save_path / "thief_qtable.npy")

    thief.save(save_path / "thief_qtable.npy")
    print(f"\nDone. Thief table saved to {save_path}/thief_qtable.npy (cop unchanged)")


if __name__ == "__main__":
    _n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    train_thief(_n)
