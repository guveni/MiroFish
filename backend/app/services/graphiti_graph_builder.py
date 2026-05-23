"""Graphiti-backed graph build service."""

import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from .graph_builder import GraphBuilderService, GraphInfo
from ..utils import graphiti_client
from ..utils.graph_paging import fetch_all_edges, fetch_all_nodes
from ..utils.locale import t


class GraphitiGraphBuilderService(GraphBuilderService):
    """GraphBuilderService implementation backed by Graphiti + Neo4j."""

    def __init__(self, api_key: Optional[str] = None):
        from ..models.task import TaskManager

        self.client = graphiti_client.get_client()
        self.task_manager = TaskManager()

    def create_graph(self, name: str) -> str:
        graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"

        async def _create():
            await graphiti_client.get_client().driver.execute_query(
                """
                MERGE (g:MiroFishGraph {graph_id: $graph_id})
                SET g.name = $name,
                    g.created_at = coalesce(g.created_at, datetime()),
                    g.description = $description
                """,
                graph_id=graph_id,
                name=name,
                description="MiroFish Social Simulation Graph",
            )

        graphiti_client.run_async(_create())
        return graph_id

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        graphiti_client.register_ontology(graph_id, ontology)

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        total_chunks = len(chunks)
        episode_uuids: List[str] = []
        ontology = graphiti_client.get_ontology(graph_id)

        for i, chunk in enumerate(chunks, 1):
            if progress_callback and total_chunks:
                progress_callback(
                    t(
                        "progress.sendingBatch",
                        current=(i + batch_size - 1) // batch_size,
                        total=(total_chunks + batch_size - 1) // batch_size,
                        chunks=1,
                    ),
                    i / total_chunks,
                )

            async def _add_episode(index=i, body=chunk):
                from graphiti_core.nodes import EpisodeType

                return await graphiti_client.get_client().add_episode(
                    name=f"chunk-{index}",
                    episode_body=body,
                    source=EpisodeType.text,
                    source_description="MiroFish Graph",
                    reference_time=graphiti_client.utcnow(),
                    group_id=graph_id,
                    entity_types=ontology.get("entity_types"),
                    edge_types=ontology.get("edge_types"),
                    edge_type_map=ontology.get("edge_type_map"),
                )

            result = graphiti_client.run_async(_add_episode())
            episode = getattr(result, "episode", None)
            episode_uuid = getattr(episode, "uuid", None)
            if episode_uuid:
                episode_uuids.append(episode_uuid)
            if i % max(batch_size, 1) == 0:
                time.sleep(0.2)

        return episode_uuids

    def _wait_for_episodes(
        self,
        episode_uuids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600,
    ):
        if progress_callback:
            progress_callback(
                t("progress.processingComplete", completed=len(episode_uuids), total=len(episode_uuids)),
                1.0,
            )

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        nodes = fetch_all_nodes(graph_id)
        edges = fetch_all_edges(graph_id)
        entity_types = set()
        for node in nodes:
            for label in getattr(node, "labels", []) or []:
                if label not in ["Entity", "Node"]:
                    entity_types.add(label)
        return GraphInfo(
            graph_id=graph_id,
            node_count=len(nodes),
            edge_count=len(edges),
            entity_types=list(entity_types),
        )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        nodes = fetch_all_nodes(graph_id)
        edges = fetch_all_edges(graph_id)
        node_map = {getattr(node, "uuid", ""): getattr(node, "name", "") or "" for node in nodes}

        nodes_data = []
        for node in nodes:
            created_at = getattr(node, "created_at", None)
            nodes_data.append(
                {
                    "uuid": getattr(node, "uuid", ""),
                    "name": getattr(node, "name", "") or "",
                    "labels": getattr(node, "labels", None) or [],
                    "summary": getattr(node, "summary", "") or "",
                    "attributes": getattr(node, "attributes", None) or {},
                    "created_at": str(created_at) if created_at else None,
                }
            )

        edges_data = []
        for edge in edges:
            created_at = getattr(edge, "created_at", None)
            valid_at = getattr(edge, "valid_at", None)
            invalid_at = getattr(edge, "invalid_at", None)
            expired_at = getattr(edge, "expired_at", None)
            episodes = getattr(edge, "episodes", None) or []
            edges_data.append(
                {
                    "uuid": getattr(edge, "uuid", ""),
                    "name": getattr(edge, "name", "") or "",
                    "fact": getattr(edge, "fact", "") or "",
                    "fact_type": getattr(edge, "name", "") or "",
                    "source_node_uuid": getattr(edge, "source_node_uuid", "") or "",
                    "target_node_uuid": getattr(edge, "target_node_uuid", "") or "",
                    "source_node_name": node_map.get(getattr(edge, "source_node_uuid", ""), ""),
                    "target_node_name": node_map.get(getattr(edge, "target_node_uuid", ""), ""),
                    "attributes": getattr(edge, "attributes", None) or {},
                    "created_at": str(created_at) if created_at else None,
                    "valid_at": str(valid_at) if valid_at else None,
                    "invalid_at": str(invalid_at) if invalid_at else None,
                    "expired_at": str(expired_at) if expired_at else None,
                    "episodes": [str(ep) for ep in episodes],
                }
            )

        return {
            "graph_id": graph_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }

    def delete_graph(self, graph_id: str):
        async def _delete():
            from graphiti_core.nodes import Node

            await Node.delete_by_group_id(graphiti_client.get_client().driver, graph_id)
            await graphiti_client.get_client().driver.execute_query(
                "MATCH (g:MiroFishGraph {graph_id: $graph_id}) DETACH DELETE g",
                graph_id=graph_id,
            )

        graphiti_client.run_async(_delete())
