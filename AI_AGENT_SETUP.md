# AI Agent Setup

This repo is configured for Codex, Cursor, and Claude Code with matching project guidance and Karpathy-inspired coding rules.

## Files

| Tool | File | Purpose |
| --- | --- | --- |
| Codex | `AGENTS.md` | Repo-level instructions loaded by Codex. |
| Cursor | `.cursor/rules/*.mdc` | Project rules loaded by Cursor. |
| Claude Code | `CLAUDE.md` | Project memory loaded by Claude Code. |
| Humans | `CURSOR.md` and this file | Setup notes with command explanations. |

## First-Time Setup

```bash
npm run setup:all
```

Installs all dependencies: root Node packages, frontend Node packages, and backend Python packages through `uv`.

```bash
cp .env.example .env
```

Creates a local environment file. Add the required LLM and Zep keys before running the app.

```bash
npm run dev
```

Starts both services. Frontend: `http://localhost:3000`. Backend API: `http://localhost:5001`.

## Daily Commands

```bash
npm run build
```

Builds the frontend and catches Vue/Vite build issues.

```bash
cd backend && uv run pytest
```

Runs backend tests when the changed area has tests.

```bash
npm run verify:gemini
```

Checks Gemini Vertex configuration when that integration is relevant.

## Karpathy-Inspired Rules

The shared agent behavior is:

1. Think before coding: state assumptions, ask when ambiguity matters, and surface tradeoffs.
2. Simplicity first: solve only the requested problem and avoid speculative abstractions.
3. Surgical changes: touch only what the task requires and avoid unrelated cleanup.
4. Goal-driven execution: define success, verify it, and report blockers clearly.

## Source Notes

- Codex uses `AGENTS.md` for durable repo guidance.
- Cursor uses project rules in `.cursor/rules/`, and can also read `AGENTS.md` for simple instruction setups.
- Claude Code uses `CLAUDE.md` as project memory.
- The four behavioral rules are adapted from the public `multica-ai/andrej-karpathy-skills` project and its Karpathy-inspired Claude/Cursor guidance.
