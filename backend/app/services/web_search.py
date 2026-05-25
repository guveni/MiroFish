"""Web search dispatcher — routes to the configured backend.

WEB_SEARCH_PROVIDER (from .env / Config):
    vertex_gemini  — Vertex Gemini + Google Search grounding (existing behaviour)
    keiro          — Keiro web search API
    none           — disabled; returns empty corpus immediately
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import Config

logger = logging.getLogger(__name__)

EMPTY_WEB_SEARCH_RESULTS = "EMPTY_WEB_SEARCH_RESULTS"
WEB_SEARCH_NOT_CONFIGURED = "WEB_SEARCH_NOT_CONFIGURED"
WEB_SEARCH_MODEL_NOT_SET = "WEB_SEARCH_MODEL_NOT_SET"

# Re-export legacy sentinel values so existing callers in graph.py keep working.
EMPTY_GEMINI_WEB_SEARCH_RESULTS = EMPTY_WEB_SEARCH_RESULTS
GEMINI_WEB_SEARCH_VERTEX_NOT_CONFIGURED = WEB_SEARCH_NOT_CONFIGURED
GEMINI_WEB_SEARCH_MODEL_NOT_SET = WEB_SEARCH_MODEL_NOT_SET


def web_search_configured() -> bool:
    """True when the active web-search provider is usable."""
    provider = Config.WEB_SEARCH_PROVIDER
    if provider == "vertex_gemini":
        from .web_search_vertex import vertex_web_search_configured
        return vertex_web_search_configured()
    if provider == "keiro":
        return bool(Config.KEIRO_API_KEY)
    return False


# Legacy alias for callers that still use the old name.
gemini_web_search_configured = web_search_configured


def resolve_grounding_model() -> str:
    """Return the model/identifier used by the active web-search backend.

    For Vertex Gemini this is the GEMINI_WEB_SEARCH_MODEL; for Keiro it is not
    model-dependent and returns a placeholder string.
    """
    provider = Config.WEB_SEARCH_PROVIDER
    if provider == "vertex_gemini":
        from .web_search_vertex import resolve_grounding_model as _resolve
        return _resolve()
    if provider == "keiro":
        return "keiro-search"
    raise ValueError(WEB_SEARCH_MODEL_NOT_SET)


def search_queries_to_corpus(
    queries: List[str],
    simulation_requirement: str = "",
    *,
    max_chars: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run web search via the configured provider and return aggregated corpus.

    Returns:
        (aggregated_text, search_metadata_aggregate)
    """
    provider = Config.WEB_SEARCH_PROVIDER

    if provider == "none":
        return "", {"provider": "none", "queries": 0}

    if provider == "vertex_gemini":
        from .web_search_vertex import search_queries_to_corpus as impl
    elif provider == "keiro":
        from .web_search_keiro import search_queries_to_corpus as impl
    else:
        raise ValueError(f"Unknown WEB_SEARCH_PROVIDER: {provider!r}")

    return impl(
        queries,
        simulation_requirement,
        max_chars=max_chars,
        progress_callback=progress_callback,
    )
