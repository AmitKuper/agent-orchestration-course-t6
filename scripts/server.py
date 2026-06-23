"""Group A MCP server — identical to hw6-common's mcp_server but uses QTableActor.

Run two instances (cop side + thief side) for a full match:
    uv run scripts/server.py --port 8001 --games-dir games/server_a
    uv run scripts/server.py --port 8002 --games-dir games/server_b

The only difference from the common server is take_turn loads the trained
QTableActor from models/ instead of creating a RandomActorBackend.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "hw6-common" / "src"))

from fastmcp import FastMCP  # noqa: E402
from game.sdk.sdk import get_state as sdk_get_state  # noqa: E402
from game.sdk.sdk import new_game as sdk_new_game  # noqa: E402
from game.sdk.sdk import state_hash as sdk_hash  # noqa: E402
from game.sdk.sdk import submit_action as sdk_submit_action  # noqa: E402
from game.wrappers.mcp_routes import _patch_state_game_id, register_routes  # noqa: E402
from game.wrappers.mcp_state import games_base, server_state  # noqa: E402

from actor_t6.qtable_actor import QTableActor  # noqa: E402

_MODELS = _ROOT / "models"
_CONFIG = _ROOT / "config" / "rl_config.json"

# Cache loaded Q-table actors so we don't reload from disk on every turn.
_actor_cache: dict[str, QTableActor] = {}


def _get_actor(role: str) -> QTableActor:
    """Return a cached loaded QTableActor for the given role.

    Falls back to RandomActorBackend if the trained model is not found.

    Args:
        role: "cop" or "thief".

    Returns:
        Loaded QTableActor with epsilon=0.
    """
    if role not in _actor_cache:
        path = _MODELS / f"{role}_qtable.npy"
        if path.exists():
            _actor_cache[role] = QTableActor.load(role=role, path=path, config_path=_CONFIG)
        else:
            from actor.random_actor import RandomActorBackend
            print(f"  [server] WARNING: no trained model for {role}; using RandomActorBackend")
            _actor_cache[role] = RandomActorBackend()  # type: ignore[assignment]
    return _actor_cache[role]


mcp = FastMCP(
    name="cop-thief-game-t6",
    instructions="Cop & Thief game engine (Group A). Use take_turn to advance the game.",
)
register_routes(mcp)


@mcp.tool()
def new_game_tool(
    game_id: str, cop_col: int, cop_row: int,
    thief_col: int, thief_row: int, seed: int,
) -> str:
    """Create a new game with pre-agreed positions.

    Args:
        game_id: The agreed game identifier.
        cop_col: Cop starting column.
        cop_row: Cop starting row.
        thief_col: Thief starting column.
        thief_row: Thief starting row.
        seed: The shared random seed (logged only).

    Returns:
        JSON {"game_id": "<id>"}.
    """
    try:
        result = sdk_new_game(
            grid_size=(5, 5),
            cop_pos=(cop_col, cop_row),
            thief_pos=(thief_col, thief_row),
            seed=seed,
            games_base=games_base(),
        )
        auto_id = result["game_id"]
        if auto_id != game_id:
            src = games_base() / auto_id
            if src.exists():
                shutil.move(str(src), str(games_base() / game_id))
        _patch_state_game_id(game_id)
        return json.dumps({"game_id": game_id})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def take_turn(game_id: str, actor: str) -> str:
    """Decide and submit one turn using the trained QTableActor.

    Gets state, calls the actor for (action, message), submits locally,
    forwards the action to the opponent server, and validates hash.

    Args:
        game_id: The active game identifier.
        actor: "cop" or "thief".

    Returns:
        JSON ActionResult extended with "hash_match" boolean.
    """
    from actor.actor_wrapper import ActorWrapper
    from game.wrappers.mcp_client import fetch_hash, send_receive_action

    try:
        obs = sdk_get_state(game_id, actor, games_base())
        backend = _get_actor(actor)
        wrapper = ActorWrapper(backend, role=actor)
        action, message = wrapper.get_action(obs)
        result = sdk_submit_action(
            game_id, actor, action, message=message, games_base=games_base()
        )
        response = asdict(result)

        if result.success:
            try:
                send_receive_action(game_id, actor, action, message)
                local_h = sdk_hash(game_id, games_base())
                opp_h = fetch_hash(game_id)
                response["hash_match"] = (local_h == opp_h)
                response["local_hash"] = local_h
                response["opponent_hash"] = opp_h
            except Exception as comm_err:
                response["comm_error"] = str(comm_err)

        return json.dumps(response)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def main() -> None:
    """Parse CLI args and start the MCP HTTP server."""
    parser = argparse.ArgumentParser(description="Cop & Thief MCP server (Group A)")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--games-dir", default="games")
    args = parser.parse_args()
    server_state["games_base"] = Path(args.games_dir)
    server_state["games_base"].mkdir(parents=True, exist_ok=True)
    print(f"  Starting Group A server on {args.host}:{args.port} | games: {args.games_dir}")
    mcp.run(transport="streamable-http", host=args.host, port=args.port, json_response=True)


if __name__ == "__main__":
    main()
