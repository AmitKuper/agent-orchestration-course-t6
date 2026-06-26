# Cop–Thief Balance Issue (Pursuer Dominance on 5×5)

## Summary

On the default 5×5 grid, the **cop (pursuer) structurally dominates the thief
(evader)**. Our Q-table cop converges to a near-optimal pursuit policy (~99–100%
win in self-play), and **no thief policy we trained can escape an optimal cop**.
This is expected for pursuit–evasion on a small bounded grid — it is a property
of the game, not a bug in the actor.

## Evidence

Self-play training (`scripts/train.py`, 200k episodes) plateaued at **cop win
≈ 99.3%**. To check whether the thief could be improved without weakening the
cop, we froze the expert cop (ε = 0) and trained the thief against it
(`scripts/train_thief.py`).

| Thief policy | vs **expert** cop (ε=0) | vs **weak** cop (ε=1, random) |
|---|---|---|
| Self-play thief (original) | **0.0%** | **74.0%** |
| Trained vs frozen expert cop (200k) | 0.0% | 71.6% |
| Continued, mixed expert/random cop (120k) | 0.0% | 69.9% |

(Win rates are the **thief's** win %, averaged over 3,000 episodes from random
start positions.)

Two facts stand out:

1. **vs an optimal cop the thief wins 0% regardless of training.** A deterministic
   (greedy) evader is always caught: the optimal pursuer refutes every fixed line
   on a 5×5 board.
2. **Dedicated anti-expert training makes the thief slightly *worse*, not better.**
   An always-losing opponent gives no learning gradient — every state looks
   equally hopeless, so the policy degenerates and loses the generalization the
   self-play thief had against weaker opponents.

## Why this happens

- **Small bounded grid + full-speed pursuer.** On 5×5 the cop can always reduce
  Chebyshev distance and corner the thief; the thief has no safe region.
- **Deterministic policy is exploitable.** A greedy Q-table maps each state to one
  action. Against an optimal cop, predictability is fatal — escaping would require
  a *mixed/stochastic* policy, which a greedy table cannot express at play time.
- **Reward shaping helps only against sub-optimal cops.** The thief's distance
  shaping (`distance_shaping_coeff`) buys real evasion against weak/random cops
  (≈74% win) but cannot overcome an optimal pursuer.

## Decision

- **Keep the cop table as-is** (it is the asset that wins series — roles alternate,
  so we want to dominate every sub-game we play as cop).
- **Ship the self-play thief** — it is the strongest available thief in every
  regime tested (ties at 0% vs expert, best vs weaker opponents, which is what
  matters against a real, non-optimal opponent cop).
- **Do not interpret the thief's 0% vs our own cop as a defect.** Against the
  external opponent's (non-optimal) cop, the thief's evasion behavior still scores.

## Implications for matches

Series outcome is driven by **cop strength**, since both teams alternate roles.
Our edge is the strong cop; the thief is expected to lose to any optimal cop
(ours or a future strong opponent) and to score mainly against weaker cops.
