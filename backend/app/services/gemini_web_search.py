"""Backward-compatible re-exports — all logic moved to web_search*.py."""

from .web_search import (  # noqa: F401
    EMPTY_GEMINI_WEB_SEARCH_RESULTS,
    GEMINI_WEB_SEARCH_MODEL_NOT_SET,
    GEMINI_WEB_SEARCH_VERTEX_NOT_CONFIGURED,
    gemini_web_search_configured,
    resolve_grounding_model,
    search_queries_to_corpus,
)
