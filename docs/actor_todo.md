# TODO — Actor (Part 5) Development

> Status legend: `[ ]` not started · `🚧` in progress · `[x]` done (with commit hash)
> Update this file in the same commit as the work it describes.
> Full design rationale in `docs/actor_plan.md`.

---

## Phase 0 — Project Scaffold

- [ ] Create `src/actor_t6/__init__.py`
- [ ] Create `tests/unit/` and `tests/integration/` directories with `__init__.py`
- [ ] Create `scripts/` directory with `__init__.py`
- [ ] Create `models/` directory with `.gitkeep` (add `models/*.npy` to `.gitignore`)
- [ ] Create `pyproject.toml` with:
  - `cop-thief-game` path dependency on `./hw6-common`
  - `actor_t6` as installable package under `src/`
  - Ruff, pytest, coverage config (mirrors `hw6-common` settings)
- [ ] Create `config/rl_config.json` with all hyperparameters (see plan §9)
- [ ] Add `.gitignore` entries: `models/*.npy`, `.venv/`, `__pycache__/`, `.env`
- [ ] Run `uv add numpy` and commit `pyproject.toml` + `uv.lock`
- [ ] Verify `uv run ruff check .` passes on empty scaffold

**Phase 0 status:** [ ] Not started

---

## Phase 1 — Action Space (`action_space.py`)

> Pure mapping layer — no game engine dependency.

- [ ] Define `ACTION_LIST`: ordered list of all 9 actions (`N NE E SE S SW W NW BARRIER`)
- [ ] `action_to_idx(action: str) -> int` — string → index
- [ ] `idx_to_action(idx: int) -> str` — index → string
- [ ] `build_illegal_mask(legal_moves: list[str]) -> np.ndarray` — bool mask, True = illegal
- [ ] Docstrings on all functions and module
- [ ] Zero Ruff violations

**Tests** (`tests/unit/test_action_space.py`):
- [ ] All 9 actions round-trip correctly (str → idx → str)
- [ ] `build_illegal_mask` marks correct indices as illegal
- [ ] `build_illegal_mask` with empty `legal_moves` → all illegal
- [ ] `build_illegal_mask` with all actions → none illegal

**Phase 1 status:** [ ] Not started

---

## Phase 2 — State Encoder (`state_encoder.py`)

> Converts `ObservationState` to a single integer index for Q-table lookup.

- [ ] `pos_to_idx(pos: tuple[int,int], grid_cols: int) -> int` — cell → 0..24
- [ ] `opp_to_idx(opp_pos: tuple[int,int] | None, grid_cols: int) -> int` — 0..25 (25 = unknown)
- [ ] `barrier_mask(my_pos, barriers, grid_cols, grid_rows) -> int` — 9-bit int from 3×3 neighbourhood
- [ ] `encode(obs: ObservationState, grid_cols: int, grid_rows: int) -> int` — combines all three
- [ ] `NUM_STATES` constant: 25 × 26 × 512 = 332,800
- [ ] Docstrings on all functions and module
- [ ] Zero Ruff violations

**Tests** (`tests/unit/test_state_encoder.py`):
- [ ] `encode` output in range `[0, NUM_STATES)`
- [ ] Two different positions produce different indices
- [ ] Opponent visible vs. not visible → different indices
- [ ] Barrier at neighbour cell sets correct bit in mask
- [ ] Corner positions (barrier neighbourhood clamps to grid)
- [ ] Round-trip consistency: same `ObservationState` always gives same index

**Phase 2 status:** [ ] Not started

---

## Phase 3 — Reward Function (`reward.py`)

> Translates `ActionResult` into a scalar reward per role.

- [ ] `compute_reward(result: ActionResult, role: str) -> float`
  - Terminal rewards: capture/thief_trapped → cop +20 / thief −20; survived/cop_trapped → cop −10 / thief +10
  - Step reward: cop −0.1 / thief +0.1 per non-terminal turn
  - Failed action (not `result.success`): both −5
- [ ] Docstrings on function and module
- [ ] Zero Ruff violations

**Tests** (`tests/unit/test_reward.py`):
- [ ] Cop reward for each `win_reason` value
- [ ] Thief reward for each `win_reason` value
- [ ] Step reward (non-terminal, successful)
- [ ] Failed action penalty for both roles
- [ ] Unknown role raises `ValueError`

**Phase 3 status:** [ ] Not started

---

## Phase 4 — Q-Table Actor (`qtable_actor.py`)

> Full `BaseActor` implementation with ε-greedy policy and Bellman updates.

- [ ] `QTableActor(BaseActor)` class with `__init__(role, config_path)`
  - Loads `rl_config.json`; initialises Q-table as `np.zeros((NUM_STATES, NUM_ACTIONS))`
  - `self.epsilon` starts at `epsilon_start`
- [ ] `get_action(obs) -> str`
  - Encode state via `state_encoder.encode()`
  - ε-greedy: random legal action or `argmax(Q[s])` with illegal mask applied
  - Stores `(state_idx, action_idx)` internally for use in `on_result()`
- [ ] `on_result(obs, action, result) -> None`
  - Compute reward via `reward.compute_reward()`
  - Encode next state
  - Apply Bellman update to Q-table
- [ ] `decay_epsilon() -> None` — `epsilon = max(epsilon_min, epsilon * epsilon_decay)`
- [ ] `save(path: str) -> None` — `np.save(path, self._q_table)`
- [ ] `load(cls, role, path, config_path) -> QTableActor` — classmethod
- [ ] Docstrings on all methods and class
- [ ] File ≤ 150 lines (extract helpers if needed)
- [ ] Zero Ruff violations

**Tests** (`tests/unit/test_qtable_actor.py`):
- [ ] `get_action` always returns a move in `obs.legal_moves`
- [ ] ε=1.0 → action is always from legal moves (random)
- [ ] ε=0.0 → action matches `argmax` of Q-row (greedy)
- [ ] `on_result` updates Q-table entry (value changes after update)
- [ ] `decay_epsilon` floors at `epsilon_min`
- [ ] `save` / `load` round-trip preserves Q-table values

**Phase 4 status:** [ ] Not started

---

## Phase 5 — Training Script (`scripts/train.py`)

> Self-play loop: CopActor vs. ThiefActor, alternating freeze schedule.

- [ ] Load `rl_config.json`
- [ ] Implement `sample_start_positions(grid_size, rng)` — random non-equal cop/thief positions
- [ ] Phase A (0–5k): train CopActor vs. random ThiefActor (ε=1.0 frozen)
- [ ] Phase B (5k–15k): train ThiefActor vs. frozen CopActor (ε=0)
- [ ] Phase C (15k–end): train both simultaneously, `decay_epsilon()` after each episode
- [ ] Log win rates to `models/training_log.json` every `checkpoint_interval` episodes
- [ ] Save Q-tables to `models/cop_qtable.npy` and `models/thief_qtable.npy`
- [ ] Print progress: episode number, win rate (last 1k), current ε
- [ ] Docstrings on all functions
- [ ] File ≤ 150 lines (split phases into helpers if needed)
- [ ] Zero Ruff violations

**Phase 5 status:** [ ] Not started

---

## Phase 6 — Integration Test

- [ ] `tests/integration/test_self_play.py`
  - Run one full game (cop vs. thief, both `QTableActor` with ε=1.0)
  - Assert every submitted action was in `obs.legal_moves`
  - Assert game terminates (no infinite loop)
  - Assert `ActionResult.game_over` is True at end
  - Assert no exceptions raised during the full game loop

**Phase 6 status:** [ ] Not started

---

## Phase 7 — Human Player (`human_player.py`)

> `BaseActor` implementation that reads moves from stdin and renders the grid.

- [ ] `HumanPlayer(BaseActor)` class with `__init__(role)`
- [ ] `get_action(obs) -> str`
  - Call `_render_grid(obs)` to print board
  - Print legal moves and prompt
  - Read stdin in a loop until input is in `obs.legal_moves` or `q`/`quit`
  - `q` / `quit` raises `SystemExit`
- [ ] `_render_grid(obs) -> None`
  - Print round number, roles, barriers remaining
  - Print column header and each row with `C`, `T`, `#`, `.`
- [ ] `on_result(obs, action, result) -> None` — print outcome message if `game_over`
- [ ] Docstrings on all methods and class
- [ ] File ≤ 150 lines
- [ ] Zero Ruff violations

**Tests** (`tests/unit/test_human_player.py`):
- [ ] `_render_grid` output contains `C`, `T`, correct round number (capture stdout)
- [ ] `get_action` with mocked stdin returns valid legal move
- [ ] `get_action` re-prompts on invalid input (mock stdin with [bad, good])
- [ ] `q` input raises `SystemExit`

**Phase 7 status:** [ ] Not started

---

## Phase 8 — Play Script (`scripts/play.py`)

> Interactive CLI: human vs. trained actor.

- [ ] Parse `--role cop|thief` argument (default: thief)
- [ ] Load trained `.npy` for the actor role; set `epsilon = 0.0`
- [ ] Sample random start positions or accept `--seed` argument
- [ ] Run game loop: human and actor take turns (cop always first)
- [ ] Print final result and score
- [ ] Docstrings on all functions
- [ ] Zero Ruff violations

**Usage:**
```bash
uv run scripts/play.py --role thief   # you play thief vs trained cop
uv run scripts/play.py --role cop     # you play cop vs trained thief
uv run scripts/play.py --role thief --seed 42
```

**Phase 8 status:** [ ] Not started

---

## Final Checklist (before submission)

- [ ] `uv run ruff check .` — zero violations
- [ ] `.venv/Scripts/python.exe -m pytest` — all tests pass, coverage ≥ 85%
- [ ] `git clone --recurse-submodules <url> && uv sync` works from scratch
- [ ] `uv run scripts/train.py` runs to completion and saves `.npy` files
- [ ] `uv run scripts/play.py --role thief` launches interactive game
- [ ] `docs/cost.md` updated with token usage for each phase
- [ ] `docs/TODO.md` (root) updated with Actor phase marked complete
