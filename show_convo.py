"""Print the agent conversation from a game log in readable format."""
import json
import sys
from pathlib import Path

_DEFAULT_LOG = "hw6-common/games/server_a/match0042/game.log"
log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(_DEFAULT_LOG)

for line in log_path.read_text().splitlines():
    entry = json.loads(line)
    t = entry.get("type")
    if t == "setup":
        print(f"=== GAME SETUP: cop={entry['cop']} thief={entry['thief']}"
              f" grid={entry['grid']} ===\n")
    elif t == "turn":
        actor = entry["actor"].upper()
        msg = entry.get("message", "")
        frm, to = entry.get("from"), entry.get("to")
        state = entry["state_after"]
        print(f"Turn {entry['turn']:2d} | {actor:5s} | {frm} -> {to}")
        print(f"         says: \"{msg}\"")
        print(f"         board: cop={state['cop']} thief={state['thief']}")
        print()
    elif t == "terminal":
        w, r = entry["winner"], entry["win_reason"]
        cop_pts = entry['scores']['cop']
        thief_pts = entry['scores']['thief']
        print(f"=== RESULT: {w} wins by {r} | rounds={entry['rounds']}"
              f" | cop={cop_pts}pts thief={thief_pts}pts ===")
