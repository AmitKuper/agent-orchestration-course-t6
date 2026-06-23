"""HumanPlayer — interactive BaseActor that reads moves from stdin.

Renders the grid after every observation and prompts for a move.
Slots directly into the same ActorWrapper / game loop as QTableActor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from actor.base_actor import BaseActor
from game.constants import COP, THIEF

if TYPE_CHECKING:
    from game.state import ActionResult, ObservationState

_SYMBOLS = {COP: "C", THIEF: "T"}


class HumanPlayer(BaseActor):
    """Interactive actor that prompts a human for each move via stdin.

    Renders the full grid before every prompt and lists legal moves.
    Invalid input is rejected and re-prompted; 'q' or 'quit' exits.
    """

    def __init__(self, role: str) -> None:
        """Initialise for a fixed role.

        Args:
            role: "cop" or "thief" — the human's role in this game.
        """
        self.role = role

    def get_action(self, obs: ObservationState) -> str:
        """Render the grid and read a legal move from stdin.

        Args:
            obs: Current observation from the game engine.

        Returns:
            A legal action string from obs.legal_moves.

        Raises:
            SystemExit: If the user types 'q' or 'quit'.
        """
        self._render(obs)
        legal = obs.legal_moves
        print(f"  Legal moves: {' '.join(legal)}")
        while True:
            raw = input("  Your move: ").strip().upper()
            if raw in ("Q", "QUIT"):
                print("  Quitting game.")
                raise SystemExit(0)
            if raw in legal:
                return raw
            print(f"  '{raw}' is not a legal move. Try: {' '.join(legal)}")

    def on_result(
        self,
        obs: ObservationState,
        action: str,
        result: ActionResult,
    ) -> None:
        """Print the outcome if the game is over.

        Args:
            obs: The observation that led to the action.
            action: The action that was submitted.
            result: The ActionResult from the game engine.
        """
        if result.game_over:
            winner_label = "You win!" if result.winner == self.role else "Actor wins!"
            print(f"\n  Game over - {result.winner} wins ({result.win_reason}). {winner_label}")

    def _render(self, obs: ObservationState) -> None:
        """Print the current grid state to stdout.

        Args:
            obs: Current observation; positions determine cell symbols.
        """
        cols = max(obs.my_pos[0], *(b[0] for b in obs.barriers), 0,
                   *([] if obs.opponent_pos is None else [obs.opponent_pos[0]])) + 1
        rows = max(obs.my_pos[1], *(b[1] for b in obs.barriers), 0,
                   *([] if obs.opponent_pos is None else [obs.opponent_pos[1]])) + 1
        cols, rows = max(cols, 5), max(rows, 5)

        cop_pos = obs.my_pos if obs.actor == COP else obs.opponent_pos
        thief_pos = obs.my_pos if obs.actor == THIEF else obs.opponent_pos
        barriers = set(map(tuple, obs.barriers))

        br = (f"  Barriers left: {obs.barriers_remaining}"
              if obs.barriers_remaining is not None else "")
        print(f"\n  Round {obs.round} | You: {obs.actor.upper()} ({_SYMBOLS[obs.actor]})"
              f"  Opponent: {'?' if obs.opponent_pos is None else str(obs.opponent_pos)}{br}")
        print("    " + " ".join(str(c) for c in range(cols)))
        for r in range(rows):
            row_str = []
            for c in range(cols):
                if cop_pos == (c, r):
                    row_str.append("C")
                elif thief_pos == (c, r):
                    row_str.append("T")
                elif (c, r) in barriers:
                    row_str.append("#")
                else:
                    row_str.append(".")
            print(f"  {r} {' '.join(row_str)}")
