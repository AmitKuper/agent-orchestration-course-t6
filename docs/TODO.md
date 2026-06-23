# TODO — Group A (Team 6)

> Track Group A progress only. For shared parts (Game / Agent / MCP / Gmail) see `hw6-common/docs/TODO.md`.

## Part 5 — Actor

### Phase 1: Scaffold
- [ ] Create `src/actor_t6/` package
- [ ] Implement `GroupAActor(BaseActor)` stub with `get_action()`
- [ ] Wire into `pyproject.toml` as installable package
- [ ] Add unit test skeleton under `tests/unit/`

### Phase 2: Strategy
- [ ] Implement cop strategy
- [ ] Implement thief strategy
- [ ] Integration test against common game engine

### Phase 3: Submission
- [ ] Verify full run with `git clone --recurse-submodules` + `uv sync`
- [ ] Update `docs/cost.md`
