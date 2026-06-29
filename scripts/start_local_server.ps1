<#
.SYNOPSIS
    Run the local Cop & Thief MCP server as a persistent daemon.

.DESCRIPTION
    Launches our team's MCP server (port 8001 by default) and leaves it running
    so it stays up across games — mirroring how the remote opponent server runs.
    Run matches against it with:

        run_match.py --local-url http://localhost:8001 --opponent-url <remote>

    which attaches to the already-running server instead of spawning and tearing
    it down each match (see run_match.py attach_mode / --local-url).

    Game-independent config (PLAYER_NAME, MCP keys, OPPONENT_MCP_URL, GMAIL_*) is
    read from hw6-common/.env automatically. This launcher only supplies the
    actor wiring (ACTOR_CLASS + per-role Q-table paths) and PYTHONPATH, which
    must be set in the process environment before the interpreter starts.

.PARAMETER Port
    Port to listen on (default 8001).
#>
param([int]$Port = 8001)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot          # repo root (HW6)
$common = Join-Path $root "hw6-common"
$models = Join-Path $root "models"

$env:PYTHONPATH = "$common\src;$root\src"
$env:ACTOR_CLASS = "actor_t6.qtable_actor.QTableActor"
$env:COP_ACTOR_TABLE = Join-Path $models "cop_qtable.npy"
$env:THIEF_ACTOR_TABLE = Join-Path $models "thief_qtable.npy"

Set-Location $common
Write-Output "[local-server] starting on port $Port (Ctrl+C to stop) — stays up across games"
uv run python -m game.wrappers.mcp_server --port $Port --games-dir "games/server_$Port"
