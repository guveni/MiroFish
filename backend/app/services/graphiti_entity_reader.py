"""Graphiti-backed entity reader."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .zep_entity_reader import EntityNode, ZepEntityReader
from ..utils import graphiti_client
from ..utils.graph_paging import fetch_all_edges, fetch_all_nodes
from ..utils.logger import get_logger

logger = get_logger("mirofish.graphiti_entity_reader")


class GraphitiEntityReader(ZepEntityReader):
    def __init__(self, api_key: Optional[str] = None):
        self.client = graphiti_client.get_client()

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        nodes = fetch_all_nodes(graph_id)
        return [
            {
                "uuid": getattr(node, "uuid", "") or "",
                "name": getattr(node, "name", "") or "",
                "labels": getattr(node, "labels", None) or [],
                "summary": getattr(node, "summary", "") or "",
                "attributes": getattr(node, "attributes", None) or {},
            }
            for node in nodes
        ]

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        edges = fetch_all_edges(graph_id)
        return [
            {
                "uuid": getattr(edge, "uuid", "") or "",
                "name": getattr(edge, "name", "") or "",
                "fact": getattr(edge, "fact", "") or "",
                "source_node_uuid": getattr(edge, "source_node_uuid", "") or "",
                "target_node_uuid": getattr(edge, "target_node_uuid", "") or "",
                "attributes": getattr(edge, "attributes", None) or {},
            }
            for edge in edges
        ]

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        async def _get():
            from graphiti_core.edges import EntityEdge

            return await EntityEdge.get_by_node_uuid(self.client.driver, node_uuid)

        try:
            edges = graphiti_client.run_async(_get())
        except Exception as exc:
            logger.warning("Failed to fetch Graphiti node edges for %s: %s", node_uuid, exc)
            return []

        return [
            {
                "uuid": getattr(edge, "uuid", "") or "",
                "name": getattr(edge, "name", "") or "",
                "fact": getattr(edge, "fact", "") or "",
                "source_node_uuid": getattr(edge, "source_node_uuid", "") or "",
                "target_node_uuid": getattr(edge, "target_node_uuid", "") or "",
                "attributes": getattr(edge, "attributes", None) or {},
            }
            for edge in edges
        ]

    def get_entity_with_context(self, graph_id: str, entity_uuid: str) -> Optional[EntityNode]:
        try:
            async def _get():
                from graphiti_core.nodes import EntityNode as GraphitiEntityNode

                return await GraphitiEntityNode.get_by_uuid(self.client.driver, entity_uuid)

            node = graphiti_client.run_async(_get())
            edges = self.get_node_edges(entity_uuid)
            node_map = {n["uuid"]: n for n in self.get_all_nodes(graph_id)}

            related_edges = []
            related_node_uuids = set()
            for edge in edges:
                if edge["source_node_uuid"] == entity_uuid:
                    related_edges.append(
                        {
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "target_node_uuid": edge["target_node_uuid"],
                        }
                    )
                    related_node_uuids.add(edge["target_node_uuid"])
                else:
                    related_edges.append(
                        {
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "source_node_uuid": edge["source_node_uuid"],
                        }
                    )
                    related_node_uuids.add(edge["source_node_uuid"])

            related_nodes = []
            for related_uuid in related_node_uuids:
                related_node = node_map.get(related_uuid)
                if related_node:
                    related_nodes.append(
                        {
                            "uuid": related_node["uuid"],
                            "name": related_node["name"],
                            "labels": related_node["labels"],
                            "summary": related_node.get("summary", ""),
                        }
                    )

            return EntityNode(
                uuid=getattr(node, "uuid", "") or "",
                name=getattr(node, "name", "") or "",
                labels=getattr(node, "labels", None) or [],
                summary=getattr(node, "summary", "") or "",
                attributes=getattr(node, "attributes", None) or {},
                related_edges=related_edges,
                related_nodes=related_nodes,
            )
        except Exception as exc:
            logger.error("Failed to fetch Graphiti entity %s: %s", entity_uuid, exc)
            return None
