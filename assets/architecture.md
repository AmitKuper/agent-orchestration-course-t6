# System Architecture Diagram

## Component Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    run_match.py  (Orchestrator)                       │
│                                                                        │
│  ┌─────────────────────┐        ┌──────────────────────────────────┐  │
│  │  Gatekeeper          │        │  ToolCaller (LLM tool-use loop)  │  │
│  │  gatekeeper.py       │        │  tool_caller.py                  │  │
│  │  · rate_limits.json  │        │  · drives actor turns            │  │
│  │  · retry / backoff   │        │  · generates NL messages         │  │
│  └──────────┬───────────┘        └──────────────────────────────────┘  │
│             │ LLM API (Anthropic / Ollama)                             │
└─────────────┼────────────────────────────────────────────────────────┘
              │ FastMCP Client  (BearerAuth / X-API-Key)
   ┌──────────▼──────────┐   MCP JSON-RPC   ┌──────────────────────┐
   │   MCP Server A       │◄────────────────►│   MCP Server B        │
   │   port 8001 / cloud  │  receive_action  │   port 8002 / cloud   │
   │   actor: Thief       │  get_hash        │   actor: Cop          │
   │                      │  propose_match   │                       │
   │  Surfaces:           │                  │  Surfaces:            │
   │  · Tools             │                  │  · Tools              │
   │  · Prompts           │                  │  · Prompts            │
   │  · Resources         │                  │  · Resources          │
   │                      │                  │                       │
   │  ┌────────────────┐  │                  │  ┌────────────────┐  │
   │  │  Game Engine   │  │                  │  │  Game Engine   │  │
   │  │  (replicated   │  │                  │  │  (replicated   │  │
   │  │   state mach.) │  │                  │  │   state mach.) │  │
   │  └────────────────┘  │                  │  └────────────────┘  │
   │  ┌────────────────┐  │                  │  ┌────────────────┐  │
   │  │  Actor Backend │  │                  │  │  Actor Backend │  │
   │  │  QTableActor   │  │                  │  │  QTableActor   │  │
   │  └────────────────┘  │                  │  └────────────────┘  │
   └──────────────────────┘                  └──────────────────────┘
              │
              ▼  (after 6 valid sub-games)
   ┌──────────────────────┐
   │  Gmail Reporter       │
   │  reporter.py          │
   │  · builds JSON report │
   │  · sends via Gmail API│
   │  · token-based OAuth  │
   └──────────────────────┘
```

## Q-Table Learning Curve

Training was run for 50,000 episodes of self-play (cop and thief both learning).
Then the thief was further trained for 30,000 episodes against the frozen expert cop.

### Cop Win Rate over Training (self-play, window=1000 episodes)

```
Win%
100 ┤                                    ╭─────────────────────
 90 ┤                              ╭─────╯
 80 ┤                         ╭───╯
 70 ┤                    ╭───╯
 60 ┤              ╭─────╯
 50 ┤         ╭───╯
 40 ┤    ╭───╯
 30 ┤╭───╯
 20 ┤╯
    └──────────────────────────────────────────────────────── Episode
    0      5k    10k   15k   20k   25k   30k   35k   40k   50k
```

Cop converges to ~97% win rate after 40,000 episodes.
Thief evasion improves from ~20% to ~35% after adversarial training against frozen cop.

## Hyperparameters

| Parameter | Cop | Thief |
|-----------|-----|-------|
| α (learning rate) | 0.1 | 0.1 |
| γ (discount) | 0.9 | 0.9 |
| ε start | 1.0 | 1.0 |
| ε end | 0.05 | 0.05 |
| Episodes | 50,000 | 30,000 |
| State space | 625 (25×25) | 625 (25×25) |
| Actions | 9 (8 dirs + BARRIER) | 8 (8 dirs) |
