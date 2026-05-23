# Using This Repository With Cursor

This repository includes Cursor project rules in `.cursor/rules/`. Open the repo root in Cursor and the rules load automatically.

## What Cursor Loads

- `.cursor/rules/english-only.mdc`: keeps new and modified project material in English.
- `.cursor/rules/mirofish-project.mdc`: explains the repo layout, commands, and verification expectations.
- `.cursor/rules/karpathy-guidelines.mdc`: adds the Karpathy-inspired agent discipline for cautious, minimal, verifiable edits.

## Slash Commands

Type `/` in Cursor chat to run project commands from `.cursor/commands/`:

- `/english-only`: translate non-English text to English in every file touched in the session (comments, logs, UI, docs). Aligns with `.cursor/rules/english-only.mdc`.

## Setup Commands

Run these from the repository root:

```bash
npm run setup:all
```

Installs root dependencies, frontend dependencies, and backend Python dependencies.

```bash
cp .env.example .env
```

Creates the local environment file. Fill in required LLM credentials before starting services. The default graph backend is Neo4j + Graphiti and `npm run dev` starts Neo4j with Docker.

```bash
npm run dev
```

Starts Neo4j, the backend, and the frontend together. The frontend runs at `http://localhost:3000`; the backend API runs at `http://localhost:5001`; Neo4j Browser runs at `http://localhost:7474`.

```bash
npm run dev:no-graph
```

Skips Neo4j management, useful when using legacy Zep Cloud with `GRAPH_BACKEND=zep`, `ZEP_API_KEY`, and `cd backend && uv sync --extra zep`. Switching graph backends requires rebuilding graphs from source documents.

## Common Checks

```bash
npm run build
```

Builds the frontend and catches many Vue/Vite regressions.

```bash
cd backend && uv run pytest
```

Runs backend tests when they exist for the changed area.

```bash
npm run verify:gemini
```

Verifies Gemini Vertex configuration when that integration is in scope.

## Maintaining Rules

When project conventions change, update `AGENTS.md`, `CLAUDE.md`, and the Cursor rules together so Codex, Claude Code, and Cursor receive the same expectations.
