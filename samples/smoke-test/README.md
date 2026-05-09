# Smoke-test sample (`samples/smoke-test`)

Use this to verify LLM access, Zep, graph build, and the UI flow **without writing your own scenario first**.

## What you need

- `.env` at the repo root with **LLM** (or Vertex ADC) + **ZEP_API_KEY**, as in the root `README.md`.
- Backend **`:5001`**, frontend **`http://localhost:3000`** (see `frontend/vite.config.js`; `npm run dev` starts both).

To **cut cost and latency** during tests, optionally add this to `.env`:

```env
OASIS_DEFAULT_MAX_ROUNDS=5
```

Restart the backend after editing `.env`.

## 1) Quick backend check

```bash
curl -s http://localhost:5001/health | python3 -m json.tool
```

Expect `{"status": "ok", "service": "MiroFish Backend"}`.

## 2) Test through the web UI

1. Start the stack: `npm run dev`.
2. Open **`http://localhost:3000`** (Vite may auto-open a browser tab).
3. Upload **`seed-micro-town.txt`** (`samples/smoke-test/seed-micro-town.txt`).
4. In the prediction / simulation-requirement box, paste the paragraph from **`simulation-requirement.txt`** (everything after the first line, or paste the whole file).
5. Proceed through ontology → graph build → simulation steps as prompted.

**Expect:** ontology JSON, a Zep-backed graph phase, then a simulation run. First-time LLM + Zep usage can take several minutes depending on quotas and region.

## 3) Optional: API-onlyontology step

From the repo root (after the backend is up):

```bash
curl -s -X POST "http://localhost:5001/api/graph/ontology/generate" \
  -F "simulation_requirement=$(cat samples/smoke-test/simulation-requirement.txt)" \
  -F "files=@samples/smoke-test/seed-micro-town.txt" \
  | python3 -m json.tool
```

Success returns `"success": true` and a `project_id`; use that ID in subsequent graph/simulation APIs or load the project in the UI.

## Troubleshooting

- **401 / auth errors:** verify LLM credentials or rerun `gcloud auth application-default login` for Vertex.
- **Zep errors:** confirm `ZEP_API_KEY` and Zep project quotas.
- **`requireFileUpload`:** ensure the multipart field name is **`files`** and the suffix is `.txt`, `.md`, or `.pdf`.
