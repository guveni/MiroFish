# MiroFish Agent Instructions

These instructions are for OpenAI Codex and other agents that read `AGENTS.md`.
Keep changes small, verifiable, and aligned with the existing MiroFish structure.

## Project Context

MiroFish is a multi-agent prediction and simulation app.

- Root: orchestration scripts, Docker setup, shared docs, and package scripts.
- `backend/`: Flask backend, Python 3.11-3.12, `uv`, simulation services, API routes, and utility modules.
- `frontend/`: Vue 3 + Vite frontend served on port 3000.
- `locales/`: product localization JSON. Use English as the authoring reference.
- `samples/`: smoke-test and example material.

## Setup And Run Commands

- `npm run setup:all`: install root dependencies, frontend dependencies, and backend Python dependencies.
- `npm run setup`: install root and frontend Node dependencies only.
- `npm run setup:backend`: run `uv sync` in `backend/`.
- `npm run dev`: start backend and frontend together.
- `npm run backend`: start the Flask backend on port 5001.
- `npm run frontend`: start the Vite frontend on port 3000.
- `npm run build`: build the frontend.
- `npm run verify:gemini`: run the Gemini Vertex verification script.

Before running the app, create `.env` from `.env.example` and fill in the required API keys. Do not commit secrets or local credentials.

## Coding Standards

- Write new and modified code, comments, prompts, logs, Markdown, and UI copy in English.
- Preserve multilingual content only when it is user data, a proper name, a cited quotation, or required locale data.
- Match nearby style before introducing a new pattern.
- Prefer existing helpers in `backend/app/utils`, `backend/app/services`, and `frontend/src/api`.
- Keep product strings out of Vue components when they belong in `locales/*.json`.
- Do not add new production dependencies without a clear need and explicit approval.

## Karpathy-Inspired Agent Discipline

These rules are adapted from the public Karpathy-inspired coding guidelines.

1. Think before coding.
   State assumptions when the task is ambiguous. Ask when the answer changes the implementation materially.
2. Prefer the simplest working change.
   Avoid speculative features, unnecessary abstraction, and configurability that was not requested.
3. Make surgical edits.
   Touch only files needed for the task. Do not refactor, reformat, or delete unrelated code.
4. Work from success criteria.
   Convert the request into a checkable outcome, then run the smallest relevant verification.

## Verification

- For frontend changes, prefer `npm run build` from the repo root when practical.
- For backend dependency or service changes, prefer `cd backend && uv run pytest` when tests exist for the touched area.
- For runtime wiring changes, use `npm run dev` only when interactive verification is needed.
- If a relevant check cannot run because of missing credentials, services, or dependencies, say exactly what blocked it.

## Completion Bar

A task is done when the requested behavior is implemented, unrelated diffs are avoided, the relevant command or inspection has been run, and remaining risk is called out clearly.
