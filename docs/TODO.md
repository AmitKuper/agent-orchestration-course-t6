# TODO — Group A (Team 6)

> Track Group A progress only. For shared parts (Game / Agent / MCP / Gmail) see `hw6-common/docs/TODO.md`.

## Part 5 — Actor ✅ Complete

### Phase 1: Scaffold ✅ Complete
- [x] Create `src/actor_t6/` package
- [x] Implement `QTableActor(BaseActor)` with `get_action()` and `on_result()`
- [x] Wire into `pyproject.toml` as installable package
- [x] Add unit test skeleton under `tests/unit/`

### Phase 2: Strategy ✅ Complete
- [x] Implement cop strategy (Q-table with Manhattan distance reward shaping)
- [x] Implement thief strategy (Q-table with evasion reward shaping)
- [x] Integration test against common game engine
- [x] 54 tests passing, 99% coverage

### Phase 3: Submission ✅ Complete
- [x] Verify full run with `git clone --recurse-submodules` + `uv sync`
- [x] Update `docs/cost.md`

---

## Shared Infrastructure Gaps — ✅ Complete

### §3.1 Turn timeout + forfeit tracking
- [x] `_actor_turn()` helper in `run_match.py` with `asyncio.wait_for(timeout)`
- [x] Forfeit on timeout, actor error, or illegal action
- [x] Technical loss after `max_consecutive_forfeits` consecutive forfeits

### §4.1 / §10 Terminal log data in sub-game report
- [x] `_read_terminal(sg_id)` reads `game.log` terminal entry
- [x] `rounds` and `barriers_used` added to sub-game result dict

### §5 Config reading from file
- [x] Created `hw6-common/config/config.json` with all game parameters
- [x] `mcp_resources.py` loads `_GAME_CONFIG` from config file (fallback to defaults)
- [x] `run_match.py` reads `grid_size`, `turn_timeout_seconds`, `max_consecutive_forfeits` from config
- [x] `setup.json` updated to reference `config/config.json`

### §11 Scientific README
- [x] Created `README.md` with DecPOMDP formal modeling, orchestration analysis, evidence

---

## Remote vs Local Match Infrastructure — ✅ Complete

- [x] Remote MCP server on Azure VM (51.4.97.97) behind nginx (port 80 → 8080)
- [x] OpenRouter adapter (`scripts/openrouter_adapter.py`) — Ollama shim → DeepSeek v3.2
- [x] OpenRouter adapter running locally (port 11500) for local LLM commentary
- [x] Gmail wired on both local and remote (absolute token paths)
- [x] State hash fixed (excludes admin fields — resolves cross-version hash mismatch)
- [x] `--time-debug` flag for per-phase performance timing
- [x] Verified end-to-end: 1 game local vs remote, email sent from both servers

---

## Documentation — ✅ Complete

- [x] `docs/PLAN.md` — architecture, ADRs, component breakdown
- [x] `docs/PRD_actor.md` — actor requirements
- [x] `docs/cost.md` — token cost tracking with real data
- [x] `docs/TODO.md` — this file
- [x] `README.md` — DecPOMDP modeling, orchestration analysis
- [x] `.env.example` — secret placeholder template
- [x] `docs/actor_plan.md` — detailed Q-table design and training plan

---

## Local Two-Server Ollama Test — ✅ Complete

- [x] Verified two local MCP servers (ports 8001 + 8002) run and play autonomously
- [x] Ollama `gemma3:4b` used for NL messages — GPU confirmed (RTX 3090 24 GB)
- [x] `students` field in JSON report populated from `PLAYER_NAMES` env var
- [x] `log` field (full turn-by-turn JSON) embedded in each `sub_games` entry of the report
- [x] Gmail reports sent automatically to GMAIL_RECIPIENT at end of series
- [x] `docs/ex06.md` + `scripts/start_local_server.ps1` committed

---

## Remaining / Lower Priority
- [ ] Sensitivity analysis notebook (`notebooks/sensitivity_analysis.ipynb`)
- [ ] Architecture diagram image in `assets/`
- [ ] Partial observation — `view_radius` Chebyshev filtering tuning
