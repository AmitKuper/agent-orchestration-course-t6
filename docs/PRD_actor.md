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

_To be defined._

## 4. Input / Output Contract

- **Input:** `ObservationState` — own position, visible barriers, opponent position if within `view_radius`.
- **Output:** one of the legal action strings in `obs.legal_moves`.
- Must never access hidden state (enforced by `partial_observation: true`).

## 5. Test Scenarios

- Returns a legal move for every observation (no illegal moves).
- Deterministic under a fixed seed (for reproducibility).
- Integration: runs a full 6-sub-game series against the random actor from common.
