# Pipeline Upgrade Prompt

**Copy and paste the text below into Cursor Composer, Codex, or another AI Agent to implement the upgrades.**

---

**Role**: You are an expert Python software engineer working on the `MiroFish` project.

**Goal**: Refactor the backend simulation pipeline to make it faster, highly reliable, resumable from failures, and capable of seamlessly switching between Azure OpenAI and Gemini.

**Context**: 
- The project has a Python Flask backend. 
- The foundation for checkpointing (`backend/app/services/run_checkpoint_store.py`) and retries (`backend/app/utils/pipeline_retry.py`) exists but is not fully integrated into the pipeline.
- The `LLMClient` (`backend/app/utils/llm_client.py`) currently supports standard OpenAI and Vertex AI.

Please implement the following four major improvements:

### 1. Speed (Concurrency & Parallelism)
- **Identify bottlenecks**: Review `backend/app/services/` (e.g., `simulation_runner.py`, `graph_builder.py`, `ontology_generator.py`) for sequential LLM calls.
- **Parallelize**: Use `concurrent.futures.ThreadPoolExecutor` to execute independent LLM calls concurrently. Alternatively, migrate the `LLMClient` to use `AsyncOpenAI` and `asyncio.gather` for non-blocking I/O.
- Ensure ThreadSafety for logging and database writes when running concurrently.

### 2. Reliability (Auto Retries)
- **Integrate `pipeline_retry.py`**: Wrap all LLM completions inside `llm_client.py` or the service layer using `run_pipeline_step`.
- Configure the retry logic to catch transient errors (e.g., HTTP 429 Rate Limits, HTTP 500/502/503/504) with an exponential backoff.
- Do not retry on validation errors (e.g., HTTP 400 or content filtering).

### 3. Recoverability (Checkpoints & Manual Retries)
- **Stage Checkpoints**: In the main pipeline (where stages like ontology generation, profile generation, and simulation run), integrate `run_checkpoint_store.py`.
- **Save State**: At the successful completion of a stage, call `checkpoint_project_stage()` or `checkpoint_simulation_stage()` to save the payload to the SQLite database.
- **Resume State**: At the beginning of a pipeline run, query the database. If a checkpoint exists for a given stage, load the payload from SQLite and skip the LLM generation for that stage.
- Add an environment variable or flag (e.g., `RESUME_FROM_CHECKPOINT=true/false`) to control whether to use checkpoints or start fresh.

### 4. Azure OpenAI & LLM Provider Switching
- **Config Updates**: Update `backend/app/config.py` and `.env.example` to include:
  - `LLM_PROVIDER` (choices: `openai`, `azure`, `vertex`)
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_API_VERSION` (e.g., `2024-02-15-preview`)
  - `AZURE_OPENAI_DEPLOYMENT` (for models like `gpt-5.5`)
- **Client Refactoring**: Modify `backend/app/utils/llm_client.py` to instantiate the appropriate client based on `LLM_PROVIDER`:
  - If `azure`, instantiate `openai.AzureOpenAI`.
  - If `vertex`, continue using the existing Google Vertex logic (`is_vertex_ai_enabled()`).
  - If `openai`, use the standard `openai.OpenAI` client.
- Ensure the interface remains identical across all 3 providers so the rest of the application requires no changes.

### 5. English-Only Codebase (Translation)
- **Translate Everything**: As you refactor, remove all Chinese (简体中文/繁体) from comments, docstrings, variable names, log messages, and inline UI strings in the codebase.
- **Natural English**: Replace it with natural, concise English. 
- **Locale Files**: If there are user-facing strings, ensure they are keyed through the i18n system (`locales/en.json`) and remove inline Chinese from Vue templates or Python fallback keys. 
- **Rule Compliance**: This enforces the project's strict `english-only.mdc` rule.

Please carefully edit the required files and provide a brief summary of how to test the checkpoints and the new Azure configuration.
---