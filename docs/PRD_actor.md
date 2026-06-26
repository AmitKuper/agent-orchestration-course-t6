# PRD — Actor (Part 5, Team-Specific)

> This document covers Group A's Actor implementation only.
> For the full project spec see `hw6-common/docs/PRD.md`.

## 1. Responsibility

The Actor is the strategic brain: given an `ObservationState` it returns one legal action string.
It extends `actor.base_actor.BaseActor` from the common package.

## 2. Interface

```python
class GroupAActor(BaseActor):
    def get_action(self, obs: ObservationState) -> str: ...
    def on_result(self, obs, action, result) -> None: ...  # optional
```

## 3. Strategy

The Actor uses **tabular Q-learning** with separate Q-tables for each role:

- **CopActor**: trained to minimise rounds-to-capture via positive reward on capture (+20) and step penalty (−0.1) that drives urgency.
- **ThiefActor**: trained to maximise survival duration via per-step reward (+0.1) and large bonus for surviving all rounds (+10).

State encoding: `(my_pos, opp_pos, barrier_mask)` → integer index over 332,800 states.
ε-greedy exploration decays from 1.0 → 0.05 over 200,000 self-play episodes.
At inference time ε = 0 (pure greedy). Q-tables persisted as `.npy` files.

## 4. Input / Output Contract

- **Input:** `ObservationState` — own position, visible barriers, opponent position if within `view_radius`.
- **Output:** one of the legal action strings in `obs.legal_moves`.
- Must never access hidden state (enforced by `partial_observation: true`).

## 5. Test Scenarios

- Returns a legal move for every observation (no illegal moves).
- Deterministic under a fixed seed (for reproducibility).
- Integration: runs a full 6-sub-game series against the random actor from common.
