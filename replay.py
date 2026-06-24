"""Interactive game replay — step through turns one at a time.

Displays the plain-text message each server sent to the other, then renders
the 5×5 board after the move.  Press Enter to advance, q+Enter to quit.

Usage:
    python replay.py <game.log>
    python replay.py hw6-common/games/server_a/match0077/game.log
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ACTOR_ICON = {"cop": "C", "thief": "T"}
_ROLE_LABEL = {"cop": "COP", "thief": "THIEF"}


def _board(cop: list[int], thief: list[int], barriers: list, cols: int, rows: int) -> str:
    """Render a cols×rows ASCII grid with C/T/X/B markers."""
    grid = [["." for _ in range(cols)] for _ in range(rows)]
    for bx, by in barriers:
        grid[by][bx] = "B"
    cx, cy = cop
    tx, ty = thief
    grid[cy][cx] = "C"
    if [tx, ty] == [cx, cy]:
        grid[ty][tx] = "X"          # capture — same cell
    else:
        grid[ty][tx] = "T"
    header = "  " + " ".join(str(c) for c in range(cols))
    rows_str = [header] + [f"{r} " + " ".join(grid[r]) for r in range(rows)]
    return "\n".join(rows_str)


def _pause(label: str = "") -> bool:
    """Print prompt and wait for Enter; return False if user typed q."""
    try:
        ans = input(f"\n  [{label}  Enter=next  q=quit] ").strip().lower()
        return ans != "q"
    except (EOFError, KeyboardInterrupt):
        return False


def replay(log_path: str) -> None:
    """Load and step through a game.log file interactively."""
    entries = [
        json.loads(line)
        for line in Path(log_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    setup = next((e for e in entries if e.get("type") == "setup"), None)
    grid: list[int] = setup.get("grid", [5, 5]) if setup else [5, 5]
    cols, rows = grid

    # -- opening screen -------------------------------------------------------
    print("\n" + "=" * 52)
    if setup:
        gid = setup.get("game_id", "?")
        print(f"  GAME  {gid}")
        print(f"  Cop starts at {setup['cop']}   Thief starts at {setup['thief']}")
        print(f"  Grid: {cols}x{rows}")
    print("=" * 52)
    if setup:
        print("\n" + _board(setup["cop"], setup["thief"], [], cols, rows))
    if not _pause("initial board"):
        return

    # -- turn loop ------------------------------------------------------------
    for entry in entries:
        if entry.get("type") == "turn":
            actor   = entry["actor"]
            label   = _ROLE_LABEL.get(actor, actor.upper())
            msg     = entry.get("message") or ""
            frm     = entry.get("from", "?")
            to      = entry.get("to", "?")
            success = entry.get("success", True)
            error   = entry.get("error") or ""
            state   = entry["state_after"]

            print("\n" + "-" * 52)
            print(f"  Turn {entry['turn']:>2}  |  {label}")
            print(f"  Move : {frm} -> {to}")
            # plain-text message sent from this server to the opponent
            if msg:
                print(f"  Says : \"{msg}\"")
            if not success:
                print(f"  REJECTED: {error}")
            print()
            print(_board(state["cop"], state["thief"],
                         state.get("barriers", []), cols, rows))

            if not _pause(f"turn {entry['turn']}"):
                return

        elif entry.get("type") == "terminal":
            winner = entry.get("winner", "?").upper()
            reason = entry.get("win_reason", "?")
            rounds = entry.get("rounds", "?")
            scores = entry.get("scores", {})
            print("\n" + "=" * 52)
            print(f"  GAME OVER  --  {winner} wins by {reason}")
            print(f"  Rounds: {rounds}")
            print(f"  Scores: Cop {scores.get('cop', 0)} pts  |  "
                  f"Thief {scores.get('thief', 0)} pts")
            print("=" * 52 + "\n")


def main() -> None:
    """Parse CLI argument and run the replay."""
    if len(sys.argv) < 2:
        default = Path("hw6-common/games/server_a") / next(
            (p.name for p in sorted(
                (Path("hw6-common/games/server_a").glob("*/game.log")),
                key=lambda p: p.stat().st_mtime, reverse=True,
            ) if p.is_file()), "match0000/game.log"
        )
        log = str(default)
        print(f"No log specified — using most recent: {log}")
    else:
        log = sys.argv[1]

    if not Path(log).exists():
        print(f"Error: {log} not found.")
        sys.exit(1)

    replay(log)


if __name__ == "__main__":
    main()
