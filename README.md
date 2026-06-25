# Cop & Thief: Dual AI-Agent Pursuit via MCP Servers

**HW6 — AI Agent Orchestration Course | University of Haifa**
**Group A (Team 6)**

---

## 1. Formal Problem Modeling — DecPOMDP

The Cop & Thief pursuit game is formalized as a **Decentralized Partially Observable Markov Decision Process (DecPOMDP)**:

$$\langle n, S, \{A_i\}, P, R, \{\Omega_i\}, O, \gamma \rangle$$

### 1.1 Components

| Symbol | Definition |
|--------|-----------|
| $n = 2$ | Two agents: Cop ($i=1$) and Thief ($i=2$) |
| $S$ | Joint state: $s = (p_c, p_t, B, k)$ — cop position $p_c \in G$, thief position $p_t \in G$, barrier set $B \subseteq G$ ($|B| \leq 5$), round counter $k \leq 25$ |
| $G$ | Grid cells $\{0,\ldots,4\}^2$ (default $5 \times 5$; parameterized via `config.json`) |
| $A_1$ | Cop actions: $\{N, NE, E, SE, S, SW, W, NW, BARRIER\}$ — 8 movement directions plus barrier placement |
| $A_2$ | Thief actions: $\{N, NE, E, SE, S, SW, W, NW\}$ — 8 movement directions (no barriers) |
| $P$ | Deterministic transition: given legal joint action $(a_1, a_2)$, next state is fully determined; illegal moves leave position unchanged |
| $R$ | Reward function encoding scoring table: capture → $(+20, +5)$; thief survives → $(+5, +10)$ |
| $\Omega_i$ | Partial observation for agent $i$: own position, barriers within Chebyshev radius $r=2$, opponent last-seen position (hidden if $d > r$) |
| $O$ | Observation function: filters $s$ to $\Omega_i$ — opponent position hidden when outside `view_radius` |
| $\gamma$ | Discount factor; tabular Q-learning uses $\gamma = 0.9$ (long-horizon pursuit/evasion) |

### 1.2 State Space Size

With a $5 \times 5$ grid, $|G| = 25$, $|B| \leq \binom{25}{5} = 53{,}130$ barrier configurations, and $k \leq 25$:

$$|S| = 25 \times 25 \times \sum_{b=0}^{5}\binom{25}{b} \times 25 \approx 17.7 \times 10^6$$

The Q-table actor uses a compact 3-tuple state representation $(p_c, p_t, k)$ with shape $25 \times 25 \times 25$, trading barrier awareness for tractability.

### 1.3 Partial Observation & Information Asymmetry

Each agent's observation $o_i = O(s, i)$ is a strict subset of the true state $s$.
The **key asymmetry**: the Cop knows its barrier count but not whether the Thief has detected them; the Thief observes barriers only within radius 2, so long-range barrier traps are invisible until proximity is reached.
This induces **belief uncertainty** on both sides — a hallmark of the DecPOMDP framework rather than a standard POMDP.

---

## 2. System Architecture

### 2.1 Component Overview

```
┌─────────────────────────────────────────────────────┐
│                  run_match.py (Orchestrator)         │
│  ┌──────────────┐          ┌──────────────────────┐  │
│  │  Gatekeeper  │          │  ToolCaller (LLM      │  │
│  │  (LLM calls) │          │  tool-use loop)       │  │
│  └──────────────┘          └──────────────────────┘  │
│          │  PRD §6: LLM lives here, not in server    │
└──────────┼──────────────────────────────────────────┘
           │ FastMCP Client (BearerAuth)
    ┌──────┴──────┐            ┌──────────────┐
    │ MCP Server A│            │ MCP Server B │
    │  port 8001  │◄──────────►│  port 8002   │
    │  (Thief)    │  HTTP REST │  (Cop)       │
    └─────────────┘            └──────────────┘
    Tools: get_state, take_action, get_actor_action
    Prompts: cop_rules / thief_rules
    Resources: game://config, game://{id}/state/{actor}
```

### 2.2 PRD §6 Compliance — LLM in Orchestrator

The MCP servers expose **tools only**; they contain no LLM calls.
The orchestrator implements the 3-step actor turn:

1. **`get_actor_action`** → Q-table backend returns action, no LLM
2. **Gatekeeper (orchestrator)** → LLM generates 1-sentence NL message
3. **`take_action`** → action + message submitted; cross-validates state hash

This satisfies the exercise §5.2 requirement: *"שרת ה-MCP הוא רכיב נפרד החושף כלים בלבד"* (the MCP server is a separate component exposing tools only).

### 2.3 Inter-Server Communication

After each `take_action`:
- Server A forwards the action to Server B via `POST /game/receive_action`
- Server B applies it to its local game engine
- Both engines compute `state_hash()` and compare via `GET /game/hash`
- **Hash mismatch** → technical loss, sub-game void, automatic re-run

---

## 3. Q-Table Actor (Part 5)

### 3.1 State Encoding

```python
state = cop_x * 125 + cop_y * 25 + thief_x * 5 + thief_y
# shape: 625 states × 9 actions
```

### 3.2 Training (Bellman Update)

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Learning rate $\alpha$ | 0.1 | Stable convergence on 5×5 grid |
| Discount $\gamma$ | 0.9 | Long-horizon pursuit/evasion |
| Exploration $\varepsilon$ (start) | 1.0 | Full exploration at start |
| Exploration $\varepsilon$ (end) | 0.05 | Residual exploration in deployment |
| Episodes | 50,000 | Sufficient for Q-table convergence |

### 3.3 Reward Shaping

| Event | Cop reward | Thief reward |
|-------|-----------|-------------|
| Capture (cop lands on thief) | +20 | −10 |
| Thief survives 25 rounds | −5 | +10 |
| Each step (step penalty) | −0.1 | +0.1 |
| Manhattan distance decrease (cop) | +0.5 | — |
| Manhattan distance increase (thief) | — | +0.3 |

---

## 4. Orchestration Challenges

### 4.1 Natural Language Without a Rigid Protocol

Agents exchange free-text messages describing intent: *"Moving northeast — cutting off escape route"*.
The Cop cannot verify whether the Thief has decoded the strategic intent; the Thief's NL messages may be deceptive or ambiguous.

**Challenge**: Parsing these messages must be robust to spelling variation, hedging language, and intentional misdirection.
**Solution**: The `agent/parser.py` uses keyword matching against the legal moves list rather than free-form NLP parsing, grounding ambiguous text in the game's constrained action vocabulary.

### 4.2 Consecutive Forfeit Detection

When an actor tool call times out (`turn_timeout_seconds = 30` from config) or returns an error, the orchestrator records a **forfeit**. After `max_consecutive_forfeits = 3` consecutive forfeits for any single actor, the sub-game is declared a **technical loss** and automatically re-run (up to 3 retries per sub-game).

### 4.3 State Divergence via Hash Validation

Without a central referee, each server independently runs the game engine. After every `take_action`, the initiating server fetches the opponent's `state_hash` and compares it to its own. A mismatch indicates a logic bug or network corruption and triggers a technical loss.

**Result from integration run** (seed=77): `hash_match=True` on every turn across 28 rounds — confirming both engines execute identical deterministic transitions from a shared seed.

### 4.4 Partial Observation Information Management

The `ObservationState` returned by `get_state` hides opponent position when Chebyshev distance exceeds `view_radius=2`. The Q-table actor must decide under uncertainty — if the opponent was last seen 3 rounds ago at position $(2,3)$, its current position is unknown.

**Key invariant**: hidden state never reaches the Actor backend. The `get_actor_action` tool calls `get_state` (which enforces the observation filter) before passing the observation to the backend.

---

## 5. Match Structure & Scoring

### 5.1 6-Sub-Game Series (PRD §4)

| Sub-game | Server A role | Server B role |
|----------|--------------|--------------|
| 1 | Thief | Cop |
| 2 | Cop | Thief |
| 3 | Thief | Cop |
| 4 | Cop | Thief |
| 5 | Thief | Cop |
| 6 | Cop | Thief |

Maximum score: 90 points ($20 \times 3$ as cop + $10 \times 3$ as thief). Minimum: 30 points.

### 5.2 Bonus Inter-Group Series (PRD §12)

For the inter-group bonus (3+3 role split):

| Sub-games | Group A role | Group B role |
|-----------|-------------|-------------|
| 1–3 | Cop | Thief |
| 4–6 | Thief | Cop |

---

## 6. Configuration

All game parameters are read from `hw6-common/config/config.json` at runtime — no hard-coded values.

```json
{
  "grid_size": [5, 5],
  "max_moves": 25,
  "num_games": 6,
  "max_barriers": 5,
  "turn_timeout_seconds": 30,
  "max_illegal_retries": 2,
  "max_consecutive_forfeits": 3,
  "scoring": {
    "cop_wins": {"cop": 20, "thief": 5},
    "thief_wins": {"cop": 5, "thief": 10}
  }
}
```

The `game://config` MCP resource exposes this file to the LLM client at runtime.

---

## 7. Evidence of Correct MCP Operation

### 7.1 MCP Protocol Surfaces Used

| Surface | Implementation | Purpose |
|---------|--------------|---------|
| **Tools** | `get_state`, `take_action`, `get_actor_action` | Game action submission and state retrieval |
| **Prompts** | `cop_rules(game_id)`, `thief_rules(game_id)` | LLM grounding with role-specific rulebook |
| **Resources** | `game://config`, `game://{id}/state/{actor}` | Live config and observation state |

### 7.2 Sample CLI Log (Integration Run, seed=77)

```
[match] seed=77 series_id=series0077 mode=actor game_type=internal
[match] waiting for servers...
[match] both servers up

[sub-game 1/6  attempt 1]  a=thief b=cop

[round 1]
  thief: {'success': True, 'game_over': False, 'hash_match': True, ...}
  cop:   {'success': True, 'game_over': False, 'hash_match': True, ...}

[round 2]
  thief: {'success': True, 'game_over': True, 'winner': 'cop',
          'win_reason': 'capture', 'hash_match': True}
  winner=cop (capture)

[series] totals: cop=90  thief=40
[match] servers stopped
```

### 7.3 Test Coverage

```
Tests: 185 passed | Coverage: 97% | Ruff: 0 violations
```

Key test modules validating MCP operation:
- `test_mcp_agent_tools.py` — `get_actor_action` returns legal moves from Q-table without LLM
- `test_mcp_resources.py` — `game://config` resource reflects `config.json` values
- `test_tool_caller.py` — LLM tool-use loop drives turns from orchestrator
- `test_mcp_tool_format.py` — Anthropic/Ollama wire format conversion

---

## 8. Setup & Running

### Prerequisites

```bash
git clone --recurse-submodules <repo-url>
cd hw6-group-a
uv sync
```

Copy `.env.example` to `.env` and configure:
```
MCP_API_KEY=your-secret-key
MCP_ALLOWED_API_KEYS=your-secret-key
ANTHROPIC_API_KEY=...          # or OLLAMA_BASE_URL for local LLM
LLM_MODEL=claude-haiku-4-5-20251001
```

### Train Q-table Actors

```bash
uv run python scripts/train.py --episodes 50000 --output models/
```

### Run a Match (Actor Mode — Q-table strategy)

```bash
uv run python hw6-common/scripts/run_match.py \
  --mode actor \
  --actor-class actor_t6.qtable_actor.QTableActor \
  --models-dir models \
  --seed 42
```

### Run a Match (LLM Mode — pure LLM strategy)

```bash
uv run python hw6-common/scripts/run_match.py --mode llm --seed 42
```

### Human vs. Actor (interactive)

```bash
uv run python scripts/play.py
# Numpad controls: 7=NW 8=N 9=NE 4=W 6=E 1=SW 2=S 3=SE 5=BARRIER
```

---

## 9. Repository Structure

```
hw6-group-a/
├── hw6-common/              ← shared submodule (Parts 1–4)
│   ├── src/game/            ← game engine, agent, MCP wrappers, Gmail
│   ├── config/
│   │   ├── config.json      ← game parameters (§10 of exercise)
│   │   ├── rate_limits.json ← LLM rate limiting
│   │   └── setup.json       ← application setup
│   └── scripts/run_match.py ← 6-sub-game orchestrator
├── src/actor_t6/            ← Part 5: Q-table actor (Group A)
│   ├── qtable_actor.py      ← QTableActor(BaseActor)
│   └── reward.py            ← reward shaping functions
├── models/                  ← trained Q-tables (cop_qtable.npy, thief_qtable.npy)
├── scripts/
│   ├── train.py             ← self-play Q-table training
│   └── play.py              ← human vs. actor (numpad controls)
├── docs/
│   ├── ex06.md              ← exercise specification
│   ├── PRD_actor.md         ← Actor PRD
│   └── TODO.md              ← Group A progress
└── README.md                ← this file
```

---

## 10. References

- Model Context Protocol specification: https://modelcontextprotocol.io/
- FastMCP library: https://gofastmcp.com/
- Watkins, C.J.C.H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3–4), 279–292.
- Bernstein, D.S., Givan, R., Immerman, N., & Zilberstein, S. (2002). The complexity of decentralized control of Markov decision processes. *Mathematics of Operations Research*, 27(4), 819–840.
- Exercise specification: `docs/ex06.md` (Dr. Yoram Segal, University of Haifa, 2026)
