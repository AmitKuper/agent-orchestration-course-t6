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

### Phase 3: Submission ✅ Complete
- [x] Verify full run with `git clone --recurse-submodules` + `uv sync`
- [x] Update `docs/cost.md`

---

## Shared Infrastructure Gaps — ✅ Complete (this session)

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

## Remaining (lower priority)
- [ ] Partial observation — `view_radius` Chebyshev filtering in `get_state`
- [ ] Deploy both MCP servers to cloud (Prefect / ngrok) with public URLs
- [ ] Bonus inter-group game (requires cloud deployment)
