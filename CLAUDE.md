# MiroFish Claude Code Instructions

Claude Code should use this file as the project memory for MiroFish. Keep the file practical and specific; update it only when a rule will help future work.

## Project Context

MiroFish is a swarm-intelligence prediction engine with a Flask backend and Vue 3 frontend.

- `backend/` contains Python services, API routes, models, scripts, and `uv` dependency management.
- `frontend/` contains the Vue 3 + Vite app and frontend API wrappers.
- `locales/` contains localization data. Author new product copy in English first.
- Root `package.json` coordinates install, dev, build, and verification commands.

## Commands

- `npm run setup:all`: install root, frontend, and backend dependencies.
- `npm run setup`: install Node dependencies for the root and frontend.
- `npm run setup:backend`: sync backend dependencies with `uv`.
- `npm run dev`: start the default Neo4j graph container, backend, and frontend together.
- `npm run dev:no-graph`: run frontend and backend without managing Neo4j, useful for `GRAPH_BACKEND=zep`.
- `npm run graph:up` / `npm run graph:down` / `npm run graph:logs`: manage the local Neo4j graph backend.
- `npm run backend`: run only the backend.
- `npm run frontend`: run only the frontend.
- `npm run build`: build the frontend.
- `npm run verify:gemini`: verify Gemini Vertex configuration.

Create `.env` from `.env.example` before running services. The default graph backend is Neo4j + Graphiti and `npm run dev` auto-starts Neo4j with Docker. Legacy Zep Cloud is opt-in via `GRAPH_BACKEND=zep`, `ZEP_API_KEY`, and `cd backend && uv sync --extra zep`; switching graph backends requires rebuilding graphs from source documents. Never expose API keys or local secrets in commits, logs, or generated docs.

## Repository Rules

- Write new and edited source, comments, Markdown, logs, prompts, and UI copy in English.
- Preserve non-English text only when it is required data, a proper noun, a quotation, or locale content.
- Follow existing Flask, Vue, Vite, and `uv` patterns instead of inventing a parallel structure.
- Keep edits scoped to the user request.
- Prefer project-local helpers before adding generic utilities.
- Ask before adding production dependencies or changing public setup behavior.

## Karpathy-Inspired Coding Discipline

Use these four rules during implementation, review, and refactoring:

1. Think before coding.
   Surface assumptions, ambiguity, and tradeoffs before committing to an approach.
2. Simplicity first.
   Build the minimum solution that satisfies the request. Remove speculative abstractions.
3. Surgical changes.
   Every edited line should connect to the task. Clean up only unused code created by the current change.
4. Goal-driven execution.
   Define what success looks like, run the relevant check, and keep working until the result is verified or blocked.

## Verification

- Frontend UI or build changes: run `npm run build` when practical.
- Backend logic changes: run focused backend tests with `cd backend && uv run pytest` when tests cover the area.
- Integration or service changes: run the smallest command that proves the wiring, and explain any missing external services.

When reporting back, summarize what changed, which checks ran, and any remaining risks.
