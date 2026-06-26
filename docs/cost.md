# Token Cost Tracking — Group A (Team 6)

> Track Actor (Part 5) costs only. Shared parts cost tracking is in `hw6-common/docs/cost.md`.

## Phase 1 — Scaffold & Q-Table Implementation

| Step | Input tokens | Output tokens | Notes |
|------|-------------|---------------|-------|
| action_space.py | ~800 | ~300 | Action index mapping + illegal mask |
| state_encoder.py | ~900 | ~350 | ObservationState → int encoder |
| reward.py | ~700 | ~250 | Shaped reward per role |
| qtable_actor.py | ~1,200 | ~500 | Full BaseActor implementation |
| Unit tests (4 modules) | ~2,000 | ~800 | TDD test suite |
| Integration test | ~600 | ~200 | Self-play full game |
| human_player.py | ~1,000 | ~400 | stdin + ASCII grid renderer |

**Phase 1 total:** ~7,200 input / ~2,800 output (~10,000 tokens)

## Phase 2 — Infrastructure & Remote Play

| Step | Input tokens | Output tokens | Notes |
|------|-------------|---------------|-------|
| run_match.py time-debug flag | ~3,000 | ~400 | Per-phase timing instrumentation |
| state_hash fix (hash mismatch) | ~2,500 | ~300 | Exclude admin fields from hash |
| mcp_server.py dotenv fix | ~1,500 | ~150 | load_dotenv in main() |
| openrouter_adapter.py | ~1,200 | ~400 | OpenRouter shim + load_dotenv |
| Remote server setup & debug | ~4,000 | ~600 | SSH, nginx, env config |
| Gmail path fix (absolute paths) | ~1,000 | ~150 | Token path resolution |
| LLM one-sentence constraint | ~800 | ~100 | Prompt engineering |

**Phase 2 total:** ~14,000 input / ~2,100 output (~16,100 tokens)

## Phase 3 — Documentation & Submission Prep

| Step | Input tokens | Output tokens | Notes |
|------|-------------|---------------|-------|
| PLAN.md | ~500 | ~600 | Architecture & planning doc |
| .env.example | ~200 | ~300 | Secret placeholder template |
| TODO.md update | ~400 | ~200 | Progress tracking |

**Phase 3 total:** ~1,100 input / ~1,100 output (~2,200 tokens)

## Cumulative Total

| Phase | Total tokens |
|-------|-------------|
| Phase 1 — Scaffold | ~10,000 |
| Phase 2 — Infrastructure | ~16,100 |
| Phase 3 — Docs | ~2,200 |
| **Grand total** | **~28,300** |

## LLM Game Commentary Cost (OpenRouter — DeepSeek v3.2)

Per the OpenRouter key usage stats (`/api/v1/auth/key`):

| Metric | Value |
|--------|-------|
| Total spend | $0.00496 |
| Model | deepseek/deepseek-v3.2 |
| Tokens per turn (avg) | ~120 in / ~17 out |
| Cost per turn | ~$0.000025 |
| Cost per 6-game series | ~$0.003 |

DeepSeek v3.2 via OpenRouter costs effectively zero for game commentary at this scale.
