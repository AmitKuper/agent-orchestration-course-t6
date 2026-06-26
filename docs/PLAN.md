# Architecture & Planning — Group A (Team 6)

> Part 5 Actor implementation plan. For shared Parts 1–4 architecture see `hw6-common/docs/`.

## 1. System Context (C4 Level 1)

```
┌────────────────────────────────────────────┐
│            Match Orchestrator              │
│         (run_match.py + Gatekeeper)        │
│   LLM commentary via Ollama / OpenRouter   │
└────────┬──────────────────────┬────────────┘
         │ MCP (HTTP)           │ MCP (HTTP)
         ▼                      ▼
┌─────────────────┐    ┌─────────────────────┐
│  Local Server   │    │   Remote Server     │
│  (port 8001)    │◄──►│  (51.4.97.97:80)    │
│  actor_t6       │    │  actor_t6 + OR adptr│
│  QTableActor    │    │  QTableActor        │
└─────────────────┘    └─────────────────────┘
```

## 2. Component Breakdown (C4 Level 2)

### 2.1 Actor Package (`src/actor_t6/`)

| Module | Responsibility |
|--------|---------------|
| `action_space.py` | Action string ↔ index mapping; illegal-move masking |
| `state_encoder.py` | `ObservationState` → integer state index for Q-table lookup |
| `reward.py` | Shaped reward from `ActionResult` per role (cop/thief) |
| `qtable_actor.py` | `QTableActor(BaseActor)` — ε-greedy policy, Bellman update, save/load |
| `human_player.py` | `HumanPlayer(BaseActor)` — stdin input, ASCII grid renderer |

### 2.2 Supporting Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `train.py` | Self-play training loop; produces `models/cop_qtable.npy` + `models/thief_qtable.npy` |
| `play.py` | Interactive human vs. trained actor CLI |
| `openrouter_adapter.py` | Ollama-compatible shim → OpenRouter API (avoids submodule changes) |

### 2.3 LLM Commentary Pipeline

```
run_match.py
  └─ Gatekeeper.call()
       ├─ ANTHROPIC_API_KEY set → Anthropic Claude directly
       └─ else → OLLAMA_BASE_URL/api/chat
              ├─ localhost:11434 → local Ollama (gemma3:4b)
              └─ localhost:11500 → openrouter_adapter.py → OpenRouter (DeepSeek v3.2)
```

## 3. Q-Table Design

### 3.1 State Encoding

| Component | Size | Encoding |
|-----------|------|---------|
| `my_pos` | 25 | `x*5 + y` → 0..24 |
| `opp_pos` | 26 | 25 cells + 25=unknown |
| `barrier_mask` | 512 | 9-bit Chebyshev-1 neighbourhood |
| **Total** | **332,800** | `my + 25*opp + 650*mask` |

### 3.2 Reward Shaping

| Event | Cop | Thief |
|-------|-----|-------|
| capture / thief_trapped | +20 | −20 |
| thief_survived / cop_trapped | −10 | +10 |
| per step | −0.1 | +0.1 |
| illegal action | −5 | −5 |

### 3.3 Hyperparameters (`config/rl_config.json`)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| episodes | 200,000 | Convergence at 200k (see §12 of actor_plan.md) |
| α (learning rate) | 0.1 | Standard tabular Q-learning |
| γ (discount) | 0.9 | Long-horizon pursuit/evasion |
| ε_start / ε_min | 1.0 / 0.05 | Full exploration → near-greedy |
| ε_decay | 0.999985 | Reaches ε_min at ~200k episodes |

## 4. Architectural Decisions

### ADR-1: Separate CopActor and ThiefActor Q-Tables
**Decision:** Two independent Q-tables, one per role.
**Rationale:** A single table gets contradictory reward signals (+20 as cop, −20 as thief for capture). Separate tables converge faster and produce cleaner role-specialised policies.

### ADR-2: OpenRouter Adapter as External Shim
**Decision:** `scripts/openrouter_adapter.py` translates Ollama-format requests to OpenRouter.
**Rationale:** The Gatekeeper in the read-only submodule speaks only Ollama or Anthropic formats. The shim avoids modifying shared code and allows cloud LLM commentary without Ollama running locally on the remote server.

### ADR-3: Orchestrator-Side LLM
**Decision:** LLM commentary generated in `run_match.py`, not inside MCP servers.
**Rationale:** Keeps MCP servers stateless and fast; LLM latency stays on the orchestrating machine; remote server focuses on game mechanics only.

### ADR-4: Replicated State Machine (No Central Referee)
**Decision:** Both MCP servers maintain independent game engines; actions exchanged, not state.
**Rationale:** Eliminates single point of failure; each server is authoritative for its local view; SHA-256 state hash after each action detects divergence.

## 5. Extension Points

- **New actor strategy**: extend `BaseActor` from `hw6-common/src/game/actor/base_actor.py`; set `ACTOR_CLASS` env var to point to it — no orchestrator changes needed.
- **New LLM backend**: implement an Ollama-compatible HTTP shim (like `openrouter_adapter.py`) and point `OLLAMA_BASE_URL` at it.
- **New game mechanics**: add to `config/config.json`; the `mechanics` dict propagates through `Game.new()`.
