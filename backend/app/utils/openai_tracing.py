"""
Optional LangSmith tracing for OpenAI-compatible clients (OpenAI SDK).

Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY. See:
https://docs.langchain.com/langsmith/trace-openai
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


def _truthy(val: Optional[str]) -> bool:
    if not val:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


def is_langsmith_openai_tracing_enabled() -> bool:
    return _truthy(os.environ.get("LANGSMITH_TRACING")) and bool(
        os.environ.get("LANGSMITH_API_KEY", "").strip()
    )


def _env_float(key: str) -> Optional[float]:
    raw = (os.environ.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def attach_completion_usage_metadata(
    completion: Any,
    *,
    model: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    After chat.completions.create, attach prompt/completion/token counts (+ optional USD)
    to the current LangSmith run when tracing is on.
    """
    if not is_langsmith_openai_tracing_enabled():
        return

    usage = getattr(completion, "usage", None)
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    total: Optional[int] = None

    if usage is not None:
        tokens_in = getattr(usage, "prompt_tokens", None)
        if tokens_in is None:
            tokens_in = getattr(usage, "input_tokens", None)
        tokens_out = getattr(usage, "completion_tokens", None)
        if tokens_out is None:
            tokens_out = getattr(usage, "output_tokens", None)
        total = getattr(usage, "total_tokens", None)

        # Some wrappers expose dict-like usage
        if hasattr(usage, "model_dump"):
            try:
                d = usage.model_dump()
                tokens_in = tokens_in if tokens_in is not None else d.get("prompt_tokens")
                tokens_out = tokens_out if tokens_out is not None else d.get("completion_tokens")
                total = total if total is not None else d.get("total_tokens")
            except Exception:
                pass

    meta: dict[str, Any] = {}
    if tokens_in is not None:
        meta["usage_prompt_tokens"] = tokens_in
    if tokens_out is not None:
        meta["usage_completion_tokens"] = tokens_out
    if total is not None:
        meta["usage_total_tokens"] = total
    if model:
        meta["llm_model"] = model

    in_per_m = _env_float("LLM_INPUT_USD_PER_1M")
    out_per_m = _env_float("LLM_OUTPUT_USD_PER_1M")
    if in_per_m is not None and tokens_in is not None:
        meta["estimated_input_cost_usd"] = round(tokens_in / 1_000_000.0 * in_per_m, 8)
    if out_per_m is not None and tokens_out is not None:
        meta["estimated_output_cost_usd"] = round(tokens_out / 1_000_000.0 * out_per_m, 8)

    est_in = meta.get("estimated_input_cost_usd")
    est_out = meta.get("estimated_output_cost_usd")
    if isinstance(est_in, (int, float)) or isinstance(est_out, (int, float)):
        meta["estimated_total_cost_usd"] = round(
            float(est_in or 0) + float(est_out or 0), 8
        )

    if extra:
        meta.update(extra)

    if not meta:
        return

    try:
        from langsmith.run_helpers import set_run_metadata

        set_run_metadata(**meta)
    except Exception as e:
        logger.debug("LangSmith metadata attach skipped: %s", e)


def wrap_openai_client(client: OpenAI) -> OpenAI:
    """
    Return a LangSmith-instrumented client when tracing is enabled; otherwise the same instance.
    """
    if not is_langsmith_openai_tracing_enabled():
        return client
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    except Exception:
        return client
