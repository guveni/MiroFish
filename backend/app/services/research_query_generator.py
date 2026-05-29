"""
根据模拟需求生成有限条 Gemini / Google Search grounding 查询用语（JSON）。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.locale import get_language_instruction
from ..utils.pipeline_retry import run_pipeline_step

logger = logging.getLogger(__name__)

RESEARCH_QUERY_SYSTEM = """You are a research assistant. Given a simulation / prediction requirement, output a small set of short search queries to gather factual context from a search index.

CRITICAL INSTRUCTION:
Your search queries must actively target and extract concrete, real-world examples, specific target entities, and niche companies/vendors related to the requirement.
For example, if the requirement asks to identify "underpriced companies" or "bottleneck providers" in an industry (e.g., semiconductor, memory, Edge AI, energy), do NOT just query generic concepts like "market potential" or "industry trends". Instead, generate queries to hunt for specific, concrete companies, micro-cap stocks, components, technical innovators, and real-world players in those fields.

Rules:
- Return ONLY valid JSON with shape: {"queries": ["...", ...]}
- Queries must be concise (each under 120 characters), self-contained, and suitable for keyword / semantic search.
- Cover distinct angles (actors, timeline, institutions, controversies) without overlapping wording.
- Use the same natural language as the user's requirement when possible."""


class ResearchQueryGenerator:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm = llm_client or LLMClient()

    def generate_queries(
        self,
        simulation_requirement: str,
        additional_context: Optional[str] = None,
        max_queries: Optional[int] = None,
    ) -> List[str]:
        cap = max_queries if max_queries is not None else Config.GEMINI_WEB_SEARCH_MAX_QUERIES
        cap = max(1, min(cap, 12))

        user_parts = [
            "## Simulation requirement\n\n" + simulation_requirement.strip(),
        ]
        if additional_context and additional_context.strip():
            user_parts.append("## Additional context\n\n" + additional_context.strip())
        user_parts.append(
            f"\n\nProduce at most {cap} queries in JSON: {{\"queries\": [...]}}\n"
            + get_language_instruction()
        )
        user_message = "\n\n".join(user_parts)

        messages = [
            {"role": "system", "content": RESEARCH_QUERY_SYSTEM},
            {"role": "user", "content": user_message},
        ]

        data: Dict[str, Any] = run_pipeline_step(
            "research_queries_llm_json",
            lambda: self._llm.chat_json(
                messages=messages, temperature=0.2, max_tokens=4096
            ),
        )
        raw = data.get("queries", [])
        if not isinstance(raw, list):
            raise ValueError("LLM returned invalid queries format")

        out: List[str] = []
        for item in raw:
            if isinstance(item, str):
                q = item.strip()
                if q:
                    out.append(q[:200])
            if len(out) >= cap:
                break

        if not out:
            raise ValueError("LLM produced no search queries")

        logger.info("Generated %d research queries for Gemini web search", len(out))
        return out
