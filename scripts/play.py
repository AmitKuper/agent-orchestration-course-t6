"""Interactive human vs. trained Q-table actor.

Usage:
    uv run scripts/play.py --role thief   # you play thief vs trained cop
    uv run scripts/play.py --role cop     # you play cop vs trained thief
    uv run scripts/play.py --role thief --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "hw6-common" / "src"))

from game.game import Game  # noqa: E402

from actor_t6.human_player import HumanPlayer  # noqa: E402
from actor_t6.qtable_actor import QTableActor  # noqa: E402

_MODELS = _ROOT / "models"
_CONFIG = _ROOT / "config" / "rl_config.json"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the play script."""
    p = argparse.ArgumentParser(description="Play Cop & Thief against a trained Q-table actor.")
    p.add_argument("--role", choices=["cop", "thief"], default="thief",
                   help="Your role (default: thief)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for start positions (default: random)")
    return p.parse_args()


def _sample_positions(
    grid_cols: int, grid_rows: int, rng: random.Random
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return random non-equal (cop_pos, thief_pos).

    Args:
        grid_cols: Grid width.
        grid_rows: Grid height.
        rng: Random instance.

    Returns:
        (cop_pos, thief_pos) tuples.
    """
    cells = [(c, r) for c in range(grid_cols) for r in range(grid_rows)]
    cop_pos = rng.choice(cells)
    thief_pos = rng.choice([p for p in cells if p != cop_pos])
    return cop_pos, thief_pos


def play() -> None:
    """Run one interactive game: human vs. trained actor."""
    args = _parse_args()
    human_role = args.role
    actor_role = "thief" if human_role == "cop" else "cop"

    model_path = _MODELS / f"{actor_role}_qtable.npy"
    if not model_path.exists():
        print(f"  No trained model found at {model_path}.")
        print("  Run: uv run scripts/train.py")
        sys.exit(1)

    actor = QTableActor.load(role=actor_role, path=model_path, config_path=_CONFIG)
    human = HumanPlayer(role=human_role)

    seed = args.seed if args.seed is not None else random.randint(0, 9999)
    rng = random.Random(seed)
    grid_size = (5, 5)
    cop_pos, thief_pos = _sample_positions(grid_size[0], grid_size[1], rng)

    game = Game.new("human-vs-actor", grid_size, cop_pos, thief_pos,
                    {"max_moves": 25, "max_barriers": 5})

    print(f"\n  === Cop & Thief ===  You: {human_role.upper()}  |  Actor: {actor_role.upper()}")
    print(f"  Seed: {seed}  |  Cop starts at {cop_pos}  |  Thief starts at {thief_pos}")
    print("  Type a direction (N NE E SE S SW W NW) or BARRIER (cop only). 'q' to quit.\n")

    # cop always moves first
    turn_order = (
        [("cop", human, True), ("thief", actor, False)]
        if human_role == "cop"
        else [("cop", actor, False), ("thief", human, True)]
    )

    done = False
    while not done:
        for role, backend, is_human in turn_order:
            obs = game.get_state(role)
            if not is_human:
                print(f"\n  [Actor ({role.upper()}) is thinking...]")
            action = backend.get_action(obs)
            if not is_human:
                print(f"  Actor plays: {action}")
            result = game.submit_action(role, action)
            backend.on_result(obs, action, result)
            if result.game_over:
                done = True
                break


if __name__ == "__main__":
    play()
