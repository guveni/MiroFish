"""
Gemini on Vertex：使用「联网检索 / Grounding with Google Search」汇总可写入图谱种子语料的正文。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from ..config import Config
from ..utils.pipeline_retry import run_pipeline_step

logger = logging.getLogger(__name__)

EMPTY_GEMINI_WEB_SEARCH_RESULTS = "EMPTY_GEMINI_WEB_SEARCH_RESULTS"
GEMINI_WEB_SEARCH_VERTEX_NOT_CONFIGURED = "GEMINI_WEB_SEARCH_VERTEX_NOT_CONFIGURED"
GEMINI_WEB_SEARCH_MODEL_NOT_SET = "GEMINI_WEB_SEARCH_MODEL_NOT_SET"


def _project_id() -> Optional[str]:
    return (
        (Config.VERTEX_AI_PROJECT_ID or "").strip()
        or (os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip()
    )


def _location() -> str:
    return (Config.VERTEX_AI_LOCATION or "").strip()


def gemini_web_search_configured() -> bool:
    """Grounding 使用 Vertex：需要项目 ID 与非空 region（与现有 Vertex/Gemini 配置一致）。"""
    return bool(_project_id()) and bool(_location())


def resolve_grounding_model() -> str:
    """联网检索必须使用 .env 中的 GEMINI_WEB_SEARCH_MODEL（Vertex GenAI model id），不得代码内默认模型。"""
    m = Config.GEMINI_WEB_SEARCH_MODEL
    if not m:
        raise ValueError(GEMINI_WEB_SEARCH_MODEL_NOT_SET)
    return m


def _response_to_text(resp: types.GenerateContentResponse) -> str:
    parts: List[str] = []
    for cand in resp.candidates or []:
        content = cand.content
        if content and content.parts:
            for p in content.parts:
                if getattr(p, "text", None):
                    parts.append(p.text)
    return "\n".join(parts).strip()


def _grounding_chunks_struct(cand: "types.Candidate", limit: int = 40) -> List[Dict[str, str]]:
    gm = cand.grounding_metadata
    if not gm or not gm.grounding_chunks:
        return []
    out: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ch in gm.grounding_chunks[:limit]:
        web = getattr(ch, "web", None)
        if not web:
            continue
        t = (web.title or "").strip()
        u = (web.uri or "").strip()
        key = (t, u)
        if u and key not in seen:
            seen.add(key)
            out.append({"title": t, "uri": u})
    return out


def _grounding_sources_lines(cand: "types.Candidate", limit: int = 15) -> List[str]:
    lines: List[str] = []
    for item in _grounding_chunks_struct(cand, limit=limit):
        t = item["title"]
        u = item["uri"]
        if t and u:
            lines.append(f"- {t} | {u}")
        elif u:
            lines.append(f"- {u}")
    return lines


def _serialize_usage(meta: Optional[types.GenerateContentResponseUsageMetadata]) -> Optional[Dict[str, Any]]:
    if meta is None:
        return None
    data: Dict[str, Any] = {}
    try:
        d = meta.model_dump(exclude_none=True)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    for attr in (
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "thoughts_token_count",
        "cached_content_token_count",
    ):
        v = getattr(meta, attr, None)
        if v is not None:
            data[attr] = v
    return data if data else None


def search_queries_to_corpus(
    queries: List[str],
    simulation_requirement: str = "",
    *,
    max_chars: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    对每个检索子问题调用 Gemini + Google Search grounding，合并为图谱/本体用语料。

    Returns:
        (aggregated_text, search_metadata_aggregate)
    """
    if not gemini_web_search_configured():
        raise ValueError(GEMINI_WEB_SEARCH_VERTEX_NOT_CONFIGURED)

    project = _project_id()
    assert project  # guarded above
    location = _location()
    budget = max_chars if max_chars is not None else Config.GEMINI_WEB_SEARCH_MAX_CHARS
    budget = max(1000, budget)

    model = resolve_grounding_model()
    client = genai.Client(vertexai=True, project=project, location=location)

    tools = [types.Tool(google_search=types.GoogleSearch())]
    gen_cfg = types.GenerateContentConfig(
        tools=tools,
        temperature=1.0,
        max_output_tokens=Config.GEMINI_WEB_SEARCH_MAX_OUTPUT_TOKENS,
    )

    blocks: List[str] = []
    rows: List[Dict[str, Any]] = []
    total_chars = 0
    all_uris = set()

    sim_ctx = (simulation_requirement or "").strip()
    sim_hint = f"\nOverall simulation need (context only):\n{sim_ctx[:1200]}\n" if sim_ctx else ""

    for q in queries:
        if total_chars >= budget:
            break

        user_text = (
            "You are gathering factual context from the web for a social-media simulation knowledge graph.\n"
            "Respond with concise bullet points: key actors (people, organizations), roles, stated relationships, "
            "events, and timelines. Prefer verifiable facts. No markdown headings.\n"
            f"{sim_hint}\n"
            f"Research focus:\n{q}\n"
        )

        try:
            resp = run_pipeline_step(
                "gemini_grounding_generate",
                lambda ut=user_text: client.models.generate_content(
                    model=model,
                    contents=ut,
                    config=gen_cfg,
                ),
            )
        except Exception as e:
            logger.warning(
                "Gemini web grounding failed for query %r after retries: %s", q, e
            )
            rows.append({"query": q, "model": model, "error": str(e), "result_block_count": 0})
            continue

        body = _response_to_text(resp)
        src_objs: List[Dict[str, str]] = []
        src_lines: List[str] = []
        usage_dict: Optional[Dict[str, Any]] = None
        if resp.candidates:
            c0 = resp.candidates[0]
            src_objs = _grounding_chunks_struct(c0)
            src_lines = _grounding_sources_lines(c0)
        usage_dict = _serialize_usage(resp.usage_metadata)

        for s in src_objs:
            uri = (s.get("uri") or "").strip()
            if uri:
                all_uris.add(uri)

        chunk_parts: List[str] = []
        if body:
            chunk_parts.append(body)
        if src_lines:
            chunk_parts.append("Sources:\n" + "\n".join(src_lines))

        chunk = "\n\n".join(chunk_parts).strip()
        meta_row: Dict[str, Any] = {
            "query": q,
            "model": model,
            "sources": src_objs,
            "usage": usage_dict,
            "result_block_count": 1 if body else 0,
        }
        rows.append(meta_row)

        if not chunk:
            continue

        if len(chunk) > 12000:
            chunk = chunk[:12000] + "\n...(trimmed)"

        remain = budget - total_chars
        if remain <= 0:
            break
        if len(chunk) > remain:
            chunk = chunk[:remain]

        blocks.append(f"### Query: {q}\n{chunk}")
        total_chars += len(chunk) + 1

    text = "\n\n---\n\n".join(blocks)
    if len(text) > budget:
        text = text[:budget] + "\n\n...(gemini web search corpus truncated)"

    if not text.strip():
        raise ValueError(EMPTY_GEMINI_WEB_SEARCH_RESULTS)

    aggregate: Dict[str, Any] = {
        "queries_detail": rows,
        "grounding_model": model,
        "query_count": len(queries),
        "unique_source_uri_count": len(all_uris),
        "web_search_chars": len(text),
    }

    logger.info(
        "Gemini Google Search corpus length=%d queries=%d model=%s uris=%d",
        len(text),
        len(queries),
        model,
        len(all_uris),
    )
    return text, aggregate
