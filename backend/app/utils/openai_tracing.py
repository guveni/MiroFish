"""
Optional LangSmith tracing for OpenAI-compatible clients (OpenAI SDK).

Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY. See:
https://docs.langchain.com/langsmith/trace-openai
"""

import os
from typing import Optional

from openai import OpenAI


def _truthy(val: Optional[str]) -> bool:
    if not val:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


def is_langsmith_openai_tracing_enabled() -> bool:
    return _truthy(os.environ.get("LANGSMITH_TRACING")) and bool(
        os.environ.get("LANGSMITH_API_KEY", "").strip()
    )


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
