"""HumanPlayer — interactive BaseActor using numpad directional keys.

Numpad layout (works with NumLock ON or OFF, no Enter needed):

    7=NW  8=N   9=NE
    4=W   5=BARRIER  6=E
    1=SW  2=S   3=SE
    q = quit

When NumLock is ON  the numpad sends ASCII digits  → mapped via _CHAR_MAP.
When NumLock is OFF the numpad sends extended codes → mapped via _SCAN_MAP.
Both modes are supported transparently.
"""

from __future__ import annotations

import msvcrt
from typing import TYPE_CHECKING

from actor.base_actor import BaseActor
from game.constants import BARRIER_ACTION, COP, STAY_ACTION, THIEF

if TYPE_CHECKING:
    from game.state import ActionResult, ObservationState

_SYMBOLS = {COP: "C", THIEF: "T"}

# NumLock ON: numpad sends ASCII digits.
_CHAR_MAP: dict[str, str] = {
    "7": "NW", "8": "N",  "9": "NE",
    "4": "W",  "5": BARRIER_ACTION, "6": "E",
    "1": "SW", "2": "S",  "3": "SE",
}

# NumLock OFF / arrow keys: msvcrt yields \xe0 then a scan byte.
_SCAN_MAP: dict[int, str] = {
    0x47: "NW", 0x48: "N",  0x49: "NE",   # Home / Up / PgUp
    0x4B: "W",  0x4C: BARRIER_ACTION, 0x4D: "E",   # Left / Clear / Right
    0x4F: "SW", 0x50: "S",  0x51: "SE",   # End  / Down / PgDn
}

_CONTROLS_COP = (
    "  Controls (numpad):  7=NW 8=N 9=NE\n"
    "                      4=W  5=BARRIER 6=E\n"
    "                      1=SW 2=S 3=SE   q=quit"
)
_CONTROLS_THIEF = (
    "  Controls (numpad):  7=NW 8=N 9=NE\n"
    "                      4=W  5=STAY  6=E\n"
    "                      1=SW 2=S 3=SE   q=quit"
)


class HumanPlayer(BaseActor):
    """Interactive actor driven by numpad single-keypress controls.

    Supports both NumLock ON (digits) and NumLock OFF (arrow scan codes).
    Unknown or illegal keys are silently ignored; the prompt stays open.
    """

    def __init__(self, role: str) -> None:
        """Initialise for a fixed role.

        Args:
            role: "cop" or "thief" — the human's role in this game.
        """
        self.role = role

    def get_action(self, obs: ObservationState) -> str:
        """Render grid and block until a legal numpad key is pressed.

        Args:
            obs: Current observation from the game engine.

        Returns:
            A legal action string from obs.legal_moves.

        Raises:
            SystemExit: If the user presses 'q'.
        """
        self._render(obs)
        print(_CONTROLS_THIEF if self.role == THIEF else _CONTROLS_COP)
        print("  Press a numpad key: ", end="", flush=True)
        legal = obs.legal_moves
        while True:
            action = self._read_action()
            # Remap BARRIER key to STAY when playing as thief.
            if action == BARRIER_ACTION and self.role == THIEF:
                action = STAY_ACTION
            if action == "QUIT":
                print("\n  Quitting game.")
                raise SystemExit(0)
            if action and action in legal:
                print(action)
                return action
            if action and action not in legal:
                print(f"\n  '{action}' not available here. Press a key: ", end="", flush=True)
            # None means unrecognised key — silently wait for next press

    def on_result(
        self,
        obs: ObservationState,
        action: str,
        result: ActionResult,
    ) -> None:
        """Print the outcome when the game ends.

        Args:
            obs: The observation that led to the action.
            action: The action that was submitted.
            result: The ActionResult from the game engine.
        """
        if result.game_over:
            label = "You win!" if result.winner == self.role else "Actor wins!"
            print(f"\n  Game over - {result.winner} wins ({result.win_reason}). {label}")

    def _read_action(self) -> str | None:
        """Read one keypress and return the action string or a sentinel.

        Returns:
            Action string (e.g. "N", "BARRIER"), "QUIT" for q/Q, or None
            if the key is not mapped to any action.
        """
        ch = msvcrt.getch()
        if ch in (b"\xe0", b"\x00"):          # extended key prefix
            scan = msvcrt.getch()[0]
            return _SCAN_MAP.get(scan)
        char = ch.decode("ascii", errors="ignore").upper()
        if char in ("Q",):
            return "QUIT"
        return _CHAR_MAP.get(char)

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
