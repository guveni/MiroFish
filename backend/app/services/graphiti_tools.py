"""Graphiti-backed report/search tools."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from .zep_tools import EdgeInfo, NodeInfo, SearchResult, ZepToolsService
from ..utils import graphiti_client
from ..utils.graph_paging import fetch_all_edges, fetch_all_nodes
from ..utils.llm_client import LLMClient
from ..utils.locale import t
from ..utils.logger import get_logger

logger = get_logger("mirofish.graphiti_tools")


class GraphitiToolsService(ZepToolsService):
    """ZepToolsService-compatible implementation backed by Graphiti."""

    def __init__(self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None):
        self.client = graphiti_client.get_client()
        self._llm_client = llm_client
        logger.info(t("console.zepToolsInitialized"))

    def _edge_dict(self, edge: Any) -> Dict[str, Any]:
        return {
            "uuid": getattr(edge, "uuid", ""),
            "name": getattr(edge, "name", "") or "",
            "fact": getattr(edge, "fact", "") or "",
            "source_node_uuid": getattr(edge, "source_node_uuid", "") or "",
            "target_node_uuid": getattr(edge, "target_node_uuid", "") or "",
        }

    def _node_dict(self, node: Any) -> Dict[str, Any]:
        return {
            "uuid": getattr(node, "uuid", ""),
            "name": getattr(node, "name", "") or "",
            "labels": getattr(node, "labels", None) or [],
            "summary": getattr(node, "summary", "") or "",
        }

    def search_graph(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ) -> SearchResult:
        from ..config import Config

        logger.info(t("console.graphSearch", graphId=graph_id, query=query[:50]))
        try:
            config = self._get_search_config(scope, limit)

            results = graphiti_client.run_async(
                self.client.search_(
                    query=query,
                    config=config,
                    group_ids=[graph_id],
                ),
                timeout=Config.GRAPHITI_SEARCH_TIMEOUT,
            )

            facts = []
            edges = []
            nodes = []

            for edge in getattr(results, "edges", []) or []:
                fact = getattr(edge, "fact", "") or ""
                if fact:
                    facts.append(fact)
                edges.append(self._edge_dict(edge))

            for node in getattr(results, "nodes", []) or []:
                nodes.append(self._node_dict(node))
                summary = getattr(node, "summary", "") or ""
                name = getattr(node, "name", "") or ""
                if summary:
                    facts.append(f"[{name}]: {summary}")

            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts),
            )
        except Exception as exc:
            logger.warning(t("console.zepSearchApiFallback", error=str(exc)))
            return self._local_search(graph_id, query, limit, scope)

    def _get_search_config(self, scope: str, limit: int):
        """Return the appropriate search config, preferring RRF when no cross-encoder is available."""
        from ..config import Config

        # When embedder is local and reranker is auto, no cross-encoder is wired;
        # fall back to RRF to avoid expensive LLM-based reranking per search.
        if Config.GRAPHITI_RERANKER == "auto" and Config.GRAPHITI_EMBEDDER == "local":
            use_cross_encoder = False
        else:
            use_cross_encoder = Config.GRAPHITI_RERANKER not in ("none", "disabled", "local", "rrf")

        if use_cross_encoder:
            from graphiti_core.search.search_config_recipes import (
                COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
                EDGE_HYBRID_SEARCH_CROSS_ENCODER,
                NODE_HYBRID_SEARCH_CROSS_ENCODER,
            )
            if scope == "nodes":
                config = copy.deepcopy(NODE_HYBRID_SEARCH_CROSS_ENCODER)
            elif scope == "both":
                config = copy.deepcopy(COMBINED_HYBRID_SEARCH_CROSS_ENCODER)
            else:
                config = copy.deepcopy(EDGE_HYBRID_SEARCH_CROSS_ENCODER)
        else:
            from graphiti_core.search.search_config_recipes import (
                COMBINED_HYBRID_SEARCH_RRF,
                EDGE_HYBRID_SEARCH_RRF,
                NODE_HYBRID_SEARCH_RRF,
            )
            if scope == "nodes":
                config = copy.deepcopy(NODE_HYBRID_SEARCH_RRF)
            elif scope == "both":
                config = copy.deepcopy(COMBINED_HYBRID_SEARCH_RRF)
            else:
                config = copy.deepcopy(EDGE_HYBRID_SEARCH_RRF)

        config.limit = limit
        return config

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        nodes = fetch_all_nodes(graph_id)
        return [
            NodeInfo(
                uuid=getattr(node, "uuid", "") or "",
                name=getattr(node, "name", "") or "",
                labels=getattr(node, "labels", None) or [],
                summary=getattr(node, "summary", "") or "",
                attributes=getattr(node, "attributes", None) or {},
            )
            for node in nodes
        ]

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        edges = fetch_all_edges(graph_id)
        result = []
        for edge in edges:
            edge_info = EdgeInfo(
                uuid=getattr(edge, "uuid", "") or "",
                name=getattr(edge, "name", "") or "",
                fact=getattr(edge, "fact", "") or "",
                source_node_uuid=getattr(edge, "source_node_uuid", "") or "",
                target_node_uuid=getattr(edge, "target_node_uuid", "") or "",
            )
            if include_temporal:
                edge_info.created_at = getattr(edge, "created_at", None)
                edge_info.valid_at = getattr(edge, "valid_at", None)
                edge_info.invalid_at = getattr(edge, "invalid_at", None)
                edge_info.expired_at = getattr(edge, "expired_at", None)
            result.append(edge_info)
        return result

    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        try:
            async def _get():
                from graphiti_core.nodes import EntityNode

                return await EntityNode.get_by_uuid(self.client.driver, node_uuid)

            node = graphiti_client.run_async(_get())
            return NodeInfo(
                uuid=getattr(node, "uuid", "") or "",
                name=getattr(node, "name", "") or "",
                labels=getattr(node, "labels", None) or [],
                summary=getattr(node, "summary", "") or "",
                attributes=getattr(node, "attributes", None) or {},
            )
        except Exception as exc:
            logger.warning("Failed to fetch Graphiti node %s: %s", node_uuid, exc)
            return None
