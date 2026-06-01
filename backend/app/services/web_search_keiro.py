"""Keiro web search backend — POST /search/content with page text extraction."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests as http_requests

from ..config import Config
from ..utils.pipeline_retry import run_pipeline_step

logger = logging.getLogger(__name__)

EMPTY_RESULTS = "EMPTY_WEB_SEARCH_RESULTS"


def _post_search(query: str) -> Dict[str, Any]:
    """Single Keiro /search/content call (retried by the caller via run_pipeline_step)."""
    url = Config.KEIRO_API_BASE_URL.rstrip("/") + Config.KEIRO_SEARCH_ENDPOINT
    headers = {
        "Authorization": f"Bearer {Config.KEIRO_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    payload = {
        "query": query,
        "maxResults": Config.KEIRO_MAX_RESULTS_PER_QUERY,
    }
    resp = http_requests.post(url, json=payload, headers=headers, timeout=Config.KEIRO_TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.json()


def search_queries_to_corpus(
    queries: List[str],
    simulation_requirement: str = "",
    *,
    max_chars: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run Keiro web search for each query and aggregate extracted page text.

    Returns:
        (aggregated_text, search_metadata_aggregate)
    """
    budget = max_chars if max_chars is not None else Config.GEMINI_WEB_SEARCH_MAX_CHARS
    budget = max(1000, budget)

    blocks: List[str] = []
    rows: List[Dict[str, Any]] = []
    total_chars = 0
    all_uris: set[str] = set()
    lock = threading.Lock()

    def _process_query(q: str):
        try:
            data = run_pipeline_step(
                "keiro_search_content",
                lambda query=q: _post_search(query),
            )
            return q, data, None
        except Exception as e:
            return q, None, e

    completed_queries = 0
    with ThreadPoolExecutor(max_workers=min(len(queries), 10)) as executor:
        future_to_q = {executor.submit(_process_query, q): q for q in queries}

        for future in as_completed(future_to_q):
            q, data, err = future.result()

            with lock:
                completed_queries += 1
                if progress_callback:
                    progress_callback(completed_queries, len(queries))

                if total_chars >= budget:
                    continue

                if err:
                    logger.warning("Keiro search failed for query %r after retries: %s", q, err)
                    rows.append({"query": q, "error": str(err), "result_block_count": 0})
                    continue

                results = data.get("results") or []
                src_objs: List[Dict[str, str]] = []
                body_parts: List[str] = []
                for r in results:
                    title = (r.get("title") or "").strip()
                    uri = (r.get("url") or "").strip()
                    content = (r.get("content") or "").strip()
                    if uri:
                        all_uris.add(uri)
                        src_objs.append({"title": title, "uri": uri})
                    snippet = content or (r.get("snippet") or r.get("description") or "").strip()
                    if snippet:
                        label = f"[{title}]({uri})" if title and uri else (title or uri or "")
                        body_parts.append(f"- {label}: {snippet[:3000]}" if label else f"- {snippet[:3000]}")

                body = "\n".join(body_parts)
                src_lines = [
                    f"- {s['title']} | {s['uri']}" if s["title"] else f"- {s['uri']}"
                    for s in src_objs
                ]
                chunk_parts: List[str] = []
                if body:
                    chunk_parts.append(body)
                if src_lines:
                    chunk_parts.append("Sources:\n" + "\n".join(src_lines))

                chunk = "\n\n".join(chunk_parts).strip()
                rows.append({
                    "query": q,
                    "sources": src_objs,
                    "result_block_count": len(results),
                })

                if not chunk:
                    continue

                if len(chunk) > 12000:
                    chunk = chunk[:12000] + "\n...(trimmed)"

                remain = budget - total_chars
                if remain <= 0:
                    continue
                if len(chunk) > remain:
                    chunk = chunk[:remain]

                blocks.append(f"### Query: {q}\n{chunk}")
                total_chars += len(chunk) + 1

    text = "\n\n---\n\n".join(blocks)
    if len(text) > budget:
        text = text[:budget] + "\n\n...(keiro web search corpus truncated)"

    if not text.strip():
        raise ValueError(EMPTY_RESULTS)

    aggregate: Dict[str, Any] = {
        "provider": "keiro",
        "queries_detail": rows,
        "query_count": len(queries),
        "unique_source_uri_count": len(all_uris),
        "web_search_chars": len(text),
    }

    logger.info(
        "Keiro search corpus length=%d queries=%d uris=%d",
        len(text),
        len(queries),
        len(all_uris),
    )
    return text, aggregate
