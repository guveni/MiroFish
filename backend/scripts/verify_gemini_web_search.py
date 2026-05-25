#!/usr/bin/env python3
"""
One-shot check: Gemini on Vertex with Google Search grounding returns text.

Uses repo-root .env (via app.config) and Application Default Credentials.

Usage (from repo root):
  cd backend && uv run python scripts/verify_gemini_web_search.py
  cd backend && uv run python scripts/verify_gemini_web_search.py --query "wildfire news California"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))


def main() -> int:
    from app.services.web_search import (
        GEMINI_WEB_SEARCH_MODEL_NOT_SET,
        GEMINI_WEB_SEARCH_VERTEX_NOT_CONFIGURED,
        gemini_web_search_configured,
        search_queries_to_corpus,
    )

    ap = argparse.ArgumentParser(description="Verify Gemini Google Search grounding on Vertex")
    ap.add_argument(
        "--query",
        default="What happened in tech news this week?",
        help="Single research-style topic to ground with Google Search",
    )
    args = ap.parse_args()

    if not gemini_web_search_configured():
        print(
            "Vertex project/location not set. Set VERTEX_AI_PROJECT_ID (or GOOGLE_CLOUD_PROJECT) "
            "and VERTEX_AI_LOCATION in repo-root .env."
        )
        return 1

    try:
        text, meta = search_queries_to_corpus(
            [args.query],
            simulation_requirement="Smoke test: simulation seed context.",
            max_chars=12000,
        )
    except ValueError as e:
        if str(e) == GEMINI_WEB_SEARCH_VERTEX_NOT_CONFIGURED:
            print("Configuration error:", e)
            return 1
        if str(e) == GEMINI_WEB_SEARCH_MODEL_NOT_SET:
            print(
                "Set GEMINI_WEB_SEARCH_MODEL in .env (Vertex GenAI id, no google/ prefix), "
                "or use a Vertex Gemini LLM_MODEL_NAME such as google/gemini-3.1-flash-lite.",
                file=sys.stderr,
            )
            return 1
        print("Search failed:", e)
        return 1
    except Exception as e:
        print("Search failed:", e)
        return 1

    print("--- metadata ---")
    print(meta)
    print("--- corpus preview ---")
    print(text[:4000])
    if len(text) > 4000:
        print(f"\n... ({len(text)} chars total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
