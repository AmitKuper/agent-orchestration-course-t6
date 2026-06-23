# CLAUDE.md — Group A (Team 6)

> **Before writing any code, read `hw6-common/docs/rules.md`. All code must comply with those rules.**

## Repository Structure

This is Group A's **private** repo. It contains only Part 5 (Actor). Parts 1–4 live in the shared
submodule.

```
hw6-group-a/
├── hw6-common/              ← git submodule — shared Parts 1–4 (read-only, never push here)
│   ├── src/
│   │   ├── game/            ← Part 1: game engine
│   │   └── actor/           ← Part 2–4 + BaseActor interface
│   ├── docs/                ← shared PRD, rules, TODO, cost (authoritative for Parts 1–4)
│   └── pyproject.toml
│
├── src/
│   └── actor_t6/            ← Part 5: Group A's Actor (team-specific, private)
│
├── docs/                    ← Group A docs (Actor only)
│   ├── PRD_actor.md         ← Actor PRD
│   ├── TODO.md              ← Group A progress tracking
│   └── cost.md              ← Group A token cost tracking
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── pyproject.toml           ← depends on hw6-common via path dep; declares actor_t6
├── uv.lock
└── CLAUDE.md                ← this file
```

## Docs: Where to Look

| What | Where |
|------|-------|
| Full project spec (PRD) | `hw6-common/docs/PRD.md` |
| Code quality rules | `hw6-common/docs/rules.md` |
| Shared parts TODO | `hw6-common/docs/TODO.md` |
| Shared parts cost | `hw6-common/docs/cost.md` |
| Actor PRD (Part 5) | `docs/PRD_actor.md` |
| Group A TODO | `docs/TODO.md` |
| Group A cost tracking | `docs/cost.md` |

## Setup

```bash
git clone --recurse-submodules <this-repo-url>
uv sync
```

## Package Management

Use `uv` exclusively — no pip, no poetry:
```
uv add <package>
uv run <command>
uv sync
```
Always commit `uv.lock` alongside `pyproject.toml`.

## Key Rules (from `hw6-common/docs/rules.md`)

- **150-line max** per file.
- Every method/class/function must have a docstring.
- No code duplication — extend `BaseActor`, don't copy it.
- Commit format: `Feature: ...` | `BugFix: ...` | `Refactor: ...` | `Docs: ...`
- Track token usage in `docs/cost.md`. Update `docs/TODO.md` after every session.
- Zero Ruff violations: `uv run ruff check .`

## Submodule Rules

- **Never push to `hw6-common/`** — it is a read-only reference to the shared repo.
- To update the submodule pointer to a newer common commit:
  ```bash
  cd hw6-common && git pull origin main && cd ..
  git add hw6-common && git commit -m "Docs: bump hw6-common submodule"
  ```

## Actor Implementation

- Extend `actor.base_actor.BaseActor` from the common package.
- Implement `get_action(obs: ObservationState) -> str`.
- Must return only moves from `obs.legal_moves`.
- Must never access hidden state (partial observation is enforced).
