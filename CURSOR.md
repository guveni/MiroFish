# Using This Repository With Cursor

This repository includes Cursor project rules in `.cursor/rules/`. Open the repo root in Cursor and the rules load automatically.

## What Cursor Loads

- `.cursor/rules/english-only.mdc`: keeps new and modified project material in English.
- `.cursor/rules/mirofish-project.mdc`: explains the repo layout, commands, and verification expectations.
- `.cursor/rules/karpathy-guidelines.mdc`: adds the Karpathy-inspired agent discipline for cautious, minimal, verifiable edits.

## Setup Commands

Run these from the repository root:

```bash
npm run setup:all
```

Installs root dependencies, frontend dependencies, and backend Python dependencies.

```bash
cp .env.example .env
```

Creates the local environment file. Fill in required API keys before starting services.

```bash
npm run dev
```

Starts the backend and frontend together. The frontend runs at `http://localhost:3000`; the backend API runs at `http://localhost:5001`.

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
