# Actor Implementation Plan — Q-Table Reinforcement Learning

> Group A, Team 6. Part 5 (Actor) — Q-table phase.
> Common interface: `actor.base_actor.BaseActor` in `hw6-common`.
> DQN is planned as a future extension and is out of scope for now.
> See §12 for training design and episode count justification.

---

## 1. Overview

Two **separate** actors are trained via self-play:

| Actor | Q-table | Role | Learns |
|-------|---------|------|--------|
| `CopActor` | `cop_qtable.npy` | Always cop | How to capture the thief |
| `ThiefActor` | `thief_qtable.npy` | Always thief | How to evade the cop |

Each actor specialises in one role only — a single actor does not learn both sides.
Both extend `BaseActor` and plug into the existing `ActorWrapper` unchanged.

The Q-table actor has two modes:
- **Training mode** (`epsilon > 0`): ε-greedy exploration, Bellman updates in `on_result()`.
- **Inference mode** (`epsilon = 0`): greedy policy, no updates.

---

## 2. Actual Game API (from `hw6-common/src/game/`)

```python
# Create a game — positions must be provided explicitly (no random_seed param)
game = Game.new(
    game_id="ep-0",
    grid_size=(5, 5),
    cop_pos=(0, 0),
    thief_pos=(4, 4),
    mechanics={"max_moves": 25, "max_barriers": 5},
)

# Get observation for an actor
obs: ObservationState = game.get_state("cop")   # NOT get_observation()

# Submit an action
result: ActionResult = game.submit_action("cop", "NE")
```

`ObservationState` fields:

| Field | Type | Notes |
|-------|------|-------|
| `actor` | `str` | "cop" or "thief" |
| `round` | `int` | Current round number |
| `my_pos` | `tuple[int,int]` | Always visible |
| `opponent_pos` | `tuple[int,int] \| None` | Full pos returned by engine (view filtering is MCP layer concern; we use full state for training) |
| `barriers` | `list[tuple[int,int]]` | All placed barriers |
| `legal_moves` | `list[str]` | Pre-computed by engine — always respect this |
| `barriers_remaining` | `int \| None` | Cop only |

---

## 3. Action Space

Actions differ by role:

| Role | Available actions | Count |
|------|------------------|-------|
| Thief | `N, NE, E, SE, S, SW, W, NW` | 8 |
| Cop | `N, NE, E, SE, S, SW, W, NW, BARRIER` | 9 |

- Fixed index order matches `DIRECTIONS` dict key order + BARRIER last.
- **Illegal action masking**: before selecting an action, set Q-values of moves not in `obs.legal_moves` to `-inf` so `argmax` never picks them.

---

## 4. State Encoding (Tabular)

Encode `ObservationState` as a single integer index for Q-table lookup.

**Components:**

| Component | Values | Encoding |
|-----------|--------|----------|
| `my_pos` | 25 cells (5×5) | `x * 5 + y` → 0..24 |
| `opponent_pos` | 25 cells + "unknown" | `x * 5 + y` → 0..24; 25 = not visible |
| `barrier_mask` | 9-bit mask of 3×3 neighbourhood around `my_pos` | 0..511 |

**State index formula:**
```
state = my_idx + 25 * opp_idx + 25 * 26 * barrier_mask
```

**Total states:** 25 × 26 × 512 = **332,800** — Q-table size: 332,800 × 9 ≈ 24 MB at float32.

**Barrier mask detail:** For each of the 8 neighbours + self cell, set bit = 1 if a barrier exists
there. Cells outside grid boundaries are treated as 0.

---

## 5. Reward Function

Shaped per role so both actors have a clear training signal.
Derived from `ActionResult.winner` and `ActionResult.win_reason`.

| Event (`win_reason`) | Cop reward | Thief reward |
|----------------------|-----------|-------------|
| `capture` | **+20** | **−20** |
| `thief_trapped` | **+20** | **−20** |
| `thief_survived` | **−10** | **+10** |
| `cop_trapped` | **−10** | **+10** |
| Each non-terminal step | **−0.1** | **+0.1** |
| Illegal / failed action | **−5** | **−5** |

Step penalties/bonuses push the cop to close in quickly and the thief to stall.

---

## 6. Q-Table Update (Bellman)

```
td_target = r + γ · max_a Q(s', a)     # 0 if done (game_over)
td_error  = td_target − Q(s, a)
Q(s, a)  += α · td_error
```

Called inside `on_result()` — the actor stores `(s, a)` from `get_action()` and completes
the update when `on_result()` delivers `(r, s', done)`.

**ε-greedy exploration:**
- `ε_start = 1.0`, `ε_min = 0.05`, decay `ε *= ε_decay` after each episode.
- At inference time set `ε = 0.0`.

**Persistence:** Q-table saved/loaded as `.npy` via `numpy.save` / `numpy.load`.

---

## 7. Self-Play Training Loop

```python
for episode in range(num_episodes):
    # Random start positions (must differ)
    cop_pos, thief_pos = sample_start_positions(grid_size, rng)
    game = Game.new(f"ep-{episode}", grid_size, cop_pos, thief_pos, mechanics)

    cop   = QTableActor(role="cop",   config=cfg)
    thief = QTableActor(role="thief", config=cfg)
    done  = False

    while not done:
        for actor, backend in [("cop", cop), ("thief", thief)]:
            obs    = game.get_state(actor)
            action = backend.get_action(obs)
            result = game.submit_action(actor, action)
            backend.on_result(obs, action, result)   # Bellman update here
            if result.game_over:
                done = True
                break

    cop.decay_epsilon()
    thief.decay_epsilon()

cop.save("models/cop_qtable.npy")
thief.save("models/thief_qtable.npy")
```

---

## 8. File Structure (Q-table scope only)

All files ≤ 150 lines (per `rules.md`).

```
src/actor_t6/
├── __init__.py          exports QTableActor, HumanPlayer
├── action_space.py      action string ↔ index; illegal mask from legal_moves list
├── state_encoder.py     ObservationState → int state index
├── reward.py            reward shaping from ActionResult for cop and thief
├── qtable_actor.py      QTableActor(BaseActor) — Q-table, ε-greedy, Bellman update
└── human_player.py      HumanPlayer(BaseActor) — stdin input, grid renderer, validation

scripts/
├── train.py             self-play training loop; saves .npy to models/
└── play.py              human vs. trained actor; loads .npy, runs interactive game

config/
└── rl_config.json       all hyperparameters (no hard-coding)

models/                  .npy Q-tables saved here (git-ignored)

tests/
├── unit/
│   ├── test_action_space.py
│   ├── test_state_encoder.py
│   ├── test_reward.py
│   └── test_qtable_actor.py
└── integration/
    └── test_self_play.py    one full game cop vs thief, asserts no illegal moves
```

---

## 9. Config (`config/rl_config.json`)

```json
{
  "grid_size": [5, 5],
  "max_moves": 25,
  "max_barriers": 5,
  "num_episodes": 200000,
  "learning_rate": 0.1,
  "discount_factor": 0.9,
  "epsilon_start": 1.0,
  "epsilon_decay": 0.999985,
  "epsilon_min": 0.05,
  "checkpoint_interval": 10000,
  "model_save_path": "models/"
}
```

---

## 10. Dependencies

```bash
uv add numpy
```

No additional deps beyond `numpy` for the Q-table phase.
`cop-thief-game` (the common package) is declared as a path dependency in `pyproject.toml`.

---

## 12. Training Design

### Why separate CopActor and ThiefActor?

A single actor learning both roles would need a larger Q-table and gets contradictory reward
signals (capture = +20 as cop, −20 as thief). Separate tables let each role converge cleanly.

---

### State space & visitation math

| Component | Size | Reasoning |
|-----------|------|-----------|
| `my_pos` | 25 | 5×5 grid |
| `opp_pos + unknown` | 26 | 25 cells + "not visible" |
| `barrier_mask` | 512 | 9-bit mask of 3×3 neighbourhood |
| **Total states** | **332,800** | 25 × 26 × 512 |
| **Effective reachable** | ~25,000 | Barriers accumulate slowly; most barrier combos never occur |

For Q-learning to converge, each **state-action pair** needs ~50–100 visits.
Effective reachable states × 9 actions × 100 visits = ~22.5M step visits needed.

Average steps per game: **~20** (cop + thief moves combined on a 5×5 grid).
Required games: 22.5M / 20 = **~1.1M games** for full theoretical coverage.

In practice, the most-visited states (near-capture, evasion corridors) converge much earlier.
Empirical rule of thumb for pursuit games this size: **good policy at 50k, strong at 200k**.

---

### Recommended episode counts

| Phase | Episodes | ε range | Purpose |
|-------|----------|---------|---------|
| Exploration warm-up | 0 – 10k | 1.0 → 0.6 | Fill Q-table with random coverage |
| Core learning | 10k – 100k | 0.6 → 0.1 | Exploit + refine strategy |
| Refinement | 100k – 200k | 0.1 → 0.05 | Lock in near-optimal policy |
| **Total (recommended)** | **200,000** | | Balances quality vs. training time |

**Minimum viable:** 50,000 episodes — decent policy, cop wins ~60% vs random thief.
**Production:** 200,000 episodes — strong policy, cop wins ~80%+ vs random thief.

---

### ε decay recalibration

The original `ε_decay = 0.995` reaches `ε_min = 0.05` in only **~600 episodes** — far too fast.
Correct values for target episode counts:

| Target episodes | ε_decay formula | ε_decay value |
|----------------|----------------|--------------|
| 50,000 | `0.05 ^ (1/50000)` | **0.999940** |
| 100,000 | `0.05 ^ (1/100000)` | **0.999970** |
| 200,000 | `0.05 ^ (1/200000)` | **0.999985** |

Update `config/rl_config.json`: use `epsilon_decay = 0.999970` for 100k or `0.999985` for 200k.

---

### Speed estimate

Each step: state encode + Q lookup + argmax + Bellman update ≈ **~15 µs** in pure Python/numpy.

| Games | Steps (×20/game) | Estimated time |
|-------|-----------------|----------------|
| 10,000 | 200k | ~3 seconds |
| 50,000 | 1M | ~15 seconds |
| 200,000 | 4M | ~60 seconds |

200k games fits in **under 2 minutes** — fast enough to iterate on hyperparameters.

---

### Self-play non-stationarity

Training both actors simultaneously creates a moving target (cop improves → thief faces harder
opponent → thief improves → cop faces harder opponent). This can cause oscillation.

**Mitigation — alternating freeze schedule:**
```
Phase A (episodes 0–5k):   train CopActor  vs. random ThiefActor
Phase B (episodes 5k–15k): train ThiefActor vs. frozen CopActor (from phase A)
Phase C (episodes 15k–...): train both simultaneously
```
This gives each actor a stable baseline before joint training begins.

---

### Convergence check

After training, measure **win rate over the last 1,000 episodes**:
- Cop win rate > 70% → CopActor converged.
- Thief survival rate > 40% → ThiefActor converged.
- If win rate is still moving by ±10% across 5k-episode windows → run more episodes.

Save a win-rate log to `models/training_log.json` for analysis.

---

## 13. Human vs. Actor Interface

A terminal-based interactive game so you can play against the trained actor and judge its quality.

### Design

- Human chooses role (cop or thief) at startup.
- The trained `QTableActor` (loaded from `.npy`) plays the opposite role.
- After every move the grid is re-drawn with the updated state.
- At game end the result and reason are shown.

### Grid Display

```
Round 3 | You: THIEF (T)  Actor: COP (C)  Barriers: # | Cop barriers left: 3

  0 1 2 3 4
0 . . . . .
1 . # . . .
2 . . T . .
3 . . . . .
4 C . . . .

Your legal moves: N NE E SE S SW W NW
Enter move: _
```

- `C` = cop, `T` = thief, `#` = barrier, `.` = empty cell
- Legal moves are always listed so you never have to guess

### Input / validation

- Human types a move string (e.g. `NE`, `BARRIER`).
- If the move is not in `legal_moves`, prompt again — no penalty, just re-ask.
- `q` or `quit` exits at any time.

### File

```
scripts/
└── play.py      CLI entry point — loads config + trained .npy, runs game loop
src/actor_t6/
└── human_player.py   HumanPlayer(BaseActor) — reads stdin, validates against legal_moves
```

`HumanPlayer` extends `BaseActor` so it slots into the same `ActorWrapper` / game loop as
the Q-table actor — no special-casing in the game loop.

### Game loop (same structure as trainer)

```python
human  = HumanPlayer(role=human_role)
actor  = QTableActor.load(role=actor_role, path=cfg["model_save_path"])
actor.epsilon = 0.0          # pure greedy — no exploration during play

while not done:
    for role, backend in turn_order:          # cop always moves first
        obs    = game.get_state(role)
        action = backend.get_action(obs)      # prints grid + prompts if human
        result = game.submit_action(role, action)
        if result.game_over:
            print_result(result)
            done = True
            break
```

### Build order addition

Slot after step 7 (trainer) in the build order:
8. `human_player.py` — stdin-reading `BaseActor`
9. `scripts/play.py` — wires `HumanPlayer` + loaded `QTableActor` into a game loop

---

## 11. Build Order

1. `action_space.py` — pure mapping, no deps
2. `state_encoder.py` — depends on `action_space` + `ObservationState`
3. `reward.py` — depends on `ActionResult` + win-reason constants
4. `qtable_actor.py` — depends on 1–3; full `BaseActor` implementation
5. `config/rl_config.json` — hyperparameters
6. Unit tests for each module (steps 1–4)
7. `scripts/train.py` — self-play training loop; saves Q-tables to `models/`
8. Integration test: one full game, assert no illegal moves selected
9. `human_player.py` — `HumanPlayer(BaseActor)` with grid renderer + stdin input
10. `scripts/play.py` — loads trained `.npy`, human picks role, interactive game loop
