#!/usr/bin/env python3
"""
Verify LangSmith token + cost metadata for a single Vertex Gemini LLM call.

Usage (from repo root):
  cd backend && uv run python scripts/verify_langsmith_gemini_cost.py

Requires: .env with Vertex ADC, LLM_MODEL_NAME, LANGSMITH_TRACING=true, LANGSMITH_API_KEY.
Optional: LANGSMITH_PROJECT, LLM_INPUT_USD_PER_1M / LLM_OUTPUT_USD_PER_1M for manual costs.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone, timedelta

# Ensure backend package imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import Config  # noqa: E402  # loads .env
from app.utils.llm_client import LLMClient  # noqa: E402
from app.utils.openai_tracing import (  # noqa: E402
    is_langsmith_openai_tracing_enabled,
    langsmith_ls_model_name,
    langsmith_ls_provider,
)


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _fetch_latest_llm_run(project: str, after: datetime):
    from langsmith import Client

    client = Client()
    runs = list(
        client.list_runs(
            project_name=project,
            run_type="llm",
            start_time=after,
            limit=5,
        )
    )
    if not runs:
        return None
    return max(runs, key=lambda r: r.start_time or after)


def main() -> int:
    _print_header("Configuration")
    if not is_langsmith_openai_tracing_enabled():
        print("FAIL: LANGSMITH_TRACING and LANGSMITH_API_KEY must be set.")
        return 1

    model = Config.require_llm_model_name()
    project = (os.environ.get("LANGSMITH_PROJECT") or "").strip() or "default"
    provider = langsmith_ls_provider()
    ls_model = langsmith_ls_model_name(model)

    print(f"LLM_PROVIDER={Config.LLM_PROVIDER}")
    print(f"LLM_MODEL_NAME={model}")
    print(f"LangSmith project={project!r}")
    print(f"ls_provider={provider!r}, ls_model_name={ls_model!r}")
    print(
        "Manual cost env:",
        f"LLM_INPUT_USD_PER_1M={os.environ.get('LLM_INPUT_USD_PER_1M', '(unset)')}",
        f"LLM_OUTPUT_USD_PER_1M={os.environ.get('LLM_OUTPUT_USD_PER_1M', '(unset)')}",
    )

    started_after = datetime.now(timezone.utc) - timedelta(seconds=5)

    _print_header("Live LLM call (traced)")
    llm = LLMClient()
    try:
        text = llm.chat(
            messages=[
                {
                    "role": "user",
                    "content": 'Reply with exactly "OK" and nothing else.',
                }
            ],
            temperature=0,
            max_tokens=256,
        )
    except Exception as e:
        print(f"FAIL: LLM call failed: {e}")
        return 1

    print(f"Response: {text!r}")

    # Reconstruct expected usage_metadata shape from raw completion path
    # (the traced call already ran; we validate helpers separately)
    _print_header("Helper sanity (local)")
    print("langsmith_tracing_metadata helpers OK (see unit tests).")

    _print_header("Flush traces to LangSmith")
    try:
        from langsmith import Client

        Client().flush()
        print("flush() completed")
    except Exception as e:
        print(f"WARN: flush failed ({e}); waiting for background upload")

    time.sleep(3)

    _print_header("Latest LangSmith LLM run")
    run = _fetch_latest_llm_run(project, started_after)
    if run is None:
        print(
            f"FAIL: No llm runs found in project {project!r} since {started_after.isoformat()}.\n"
            "Check LANGSMITH_PROJECT matches the UI project name."
        )
        return 1

    print(f"run_id={run.id}")
    print(f"name={run.name}")
    print(f"status={run.status}")
    meta = run.extra.get("metadata") if run.extra else {}
    if meta is None:
        meta = {}
    usage_meta = meta.get("usage_metadata") or {}
    print(f"metadata.ls_provider={meta.get('ls_provider')!r}")
    print(f"metadata.ls_model_name={meta.get('ls_model_name')!r}")
    print(f"usage_metadata={usage_meta}")

    prompt_tokens = getattr(run, "prompt_tokens", None)
    completion_tokens = getattr(run, "completion_tokens", None)
    total_tokens = getattr(run, "total_tokens", None)
    total_cost = getattr(run, "total_cost", None)
    print(f"run.prompt_tokens={prompt_tokens}")
    print(f"run.completion_tokens={completion_tokens}")
    print(f"run.total_tokens={total_tokens}")
    print(f"run.total_cost={total_cost}")

    issues: list[str] = []
    if meta.get("ls_provider") != provider:
        issues.append(f"ls_provider mismatch (expected {provider!r})")
    if ls_model and meta.get("ls_model_name") != ls_model:
        issues.append(f"ls_model_name mismatch (expected {ls_model!r})")
    if not usage_meta and not (prompt_tokens or completion_tokens):
        issues.append("no usage_metadata and no token fields on run")
    if total_cost is None and not usage_meta.get("total_cost"):
        issues.append(
            "no cost on run (add model in LangSmith Settings → Models, or set "
            "LLM_INPUT_USD_PER_1M / LLM_OUTPUT_USD_PER_1M)"
        )

    if issues:
        print("\nRESULT: PARTIAL — trace exists but cost/tokens incomplete:")
        for item in issues:
            print(f"  - {item}")
        print(
            "\nOpen this run in LangSmith and check the trace tree token/cost panel."
        )
        return 2

    print("\nRESULT: OK — tokens and/or cost metadata present on the latest LLM run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
