"""Graphiti-backed graph build service."""

import asyncio
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional

from ..config import Config
from .graph_builder import GraphBuilderService, GraphInfo
from ..utils import graphiti_client
from ..utils.graph_paging import fetch_all_edges, fetch_all_nodes
from ..utils.locale import t
from ..utils.pipeline_retry import run_pipeline_step_async


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
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        total_chunks = len(chunks)
        if not total_chunks:
            return []

        ontology = graphiti_client.get_ontology(graph_id)
        if batch_size is None:
            batch_size = Config.GRAPHITI_BATCH_SIZE
        batch_size = max(1, batch_size)

        async def _add_all_episodes() -> List[str]:
            from graphiti_core.nodes import EpisodeType
            from graphiti_core.utils.bulk_utils import RawEpisode
            from ..utils.logger import get_logger

            build_logger = get_logger("mirofish.build")
            reference_time = graphiti_client.utcnow()
            total_batches = (total_chunks + batch_size - 1) // batch_size
            concurrency = Config.graphiti_effective_semaphore_limit()
            build_logger.info(
                "Graphiti bulk ingest: %d chunks in %d batches (concurrency=%d, provider=%s)",
                total_chunks,
                total_batches,
                concurrency,
                Config.LLM_PROVIDER,
            )

            semaphore = asyncio.Semaphore(concurrency)
            completed_batches = 0
            active_batches = set()
            progress_lock = threading.Lock()

            def update_progress(msg: str, ratio: float):
                if progress_callback:
                    progress_callback(msg, ratio)

            async def _process_batch(batch_index: int, batch_num: int) -> List[str]:
                nonlocal completed_batches
                
                batch_chunks = chunks[batch_index : batch_index + batch_size]
                
                bulk_episodes = [
                    RawEpisode(
                        name=f"chunk-{batch_index + offset + 1}",
                        content=chunk,
                        source=EpisodeType.text,
                        source_description="MiroFish Graph",
                        reference_time=reference_time,
                    )
                    for offset, chunk in enumerate(batch_chunks)
                ]

                async def _add_batch(eps=bulk_episodes):
                    return await graphiti_client.get_client().add_episode_bulk(
                        bulk_episodes=eps,
                        group_id=graph_id,
                        entity_types=ontology.get("entity_types"),
                        edge_types=ontology.get("edge_types"),
                        edge_type_map=ontology.get("edge_type_map"),
                    )

                build_logger.info(f"Batch {batch_num}/{total_batches} queuing... completed: {completed_batches}/{total_batches}.")
                with progress_lock:
                    active_batches.add(batch_num)
                    ratio = completed_batches / total_batches
                    update_progress(
                        f"Queued batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks). Completed: {completed_batches}/{total_batches}.",
                        ratio,
                    )

                heartbeat_stop = asyncio.Event()

                async def _batch_heartbeat() -> None:
                    elapsed = 0
                    interval = max(15, Config.GRAPHITI_BATCH_HEARTBEAT_SECONDS)
                    while not heartbeat_stop.is_set():
                        await asyncio.sleep(interval)
                        if heartbeat_stop.is_set():
                            break
                        elapsed += interval
                        with progress_lock:
                            ratio = completed_batches / total_batches
                            update_progress(
                                f"Processing batch {batch_num}/{total_batches} "
                                f"(LLM extraction running, ~{elapsed}s)... "
                                f"Completed: {completed_batches}/{total_batches}.",
                                ratio,
                            )

                try:
                    async with semaphore:
                        build_logger.info(f"Batch {batch_num}/{total_batches} acquired semaphore slot, starting ingest...")
                        with progress_lock:
                            ratio = completed_batches / total_batches
                            update_progress(
                                f"Processing batch {batch_num}/{total_batches} (acquired slot)... Completed: {completed_batches}/{total_batches}.",
                                ratio,
                            )

                        heartbeat_task = asyncio.create_task(_batch_heartbeat())
                        try:
                            result = await run_pipeline_step_async(
                                f"graphiti_add_episode_bulk_{batch_num}",
                                _add_batch,
                            )
                        finally:
                            heartbeat_stop.set()
                            heartbeat_task.cancel()
                            try:
                                await heartbeat_task
                            except asyncio.CancelledError:
                                pass
                        
                        with progress_lock:
                            completed_batches += 1
                            active_batches.discard(batch_num)
                            ratio = completed_batches / total_batches
                            build_logger.info(f"Batch {batch_num}/{total_batches} completed successfully. Total completed: {completed_batches}/{total_batches}.")
                            update_progress(
                                t(
                                    "progress.sendingBatch",
                                    current=completed_batches,
                                    total=total_batches,
                                    chunks=len(batch_chunks),
                                ) + f" (Active: {len(active_batches)} batches)",
                                ratio,
                            )
                            
                        uuids = []
                        for episode in getattr(result, "episodes", []) or []:
                            episode_uuid = getattr(episode, "uuid", None)
                            if episode_uuid:
                                uuids.append(str(episode_uuid))
                        return uuids
                except Exception as e:
                    build_logger.error(f"Batch {batch_num}/{total_batches} failed: {e}")
                    with progress_lock:
                        active_batches.discard(batch_num)
                    raise

            tasks = []
            for batch_index in range(0, total_chunks, batch_size):
                batch_num = batch_index // batch_size + 1
                tasks.append(asyncio.create_task(_process_batch(batch_index, batch_num)))
                
            results = await asyncio.gather(*tasks)
            
            episode_uuids = []
            for batch_uuids in results:
                episode_uuids.extend(batch_uuids)
                
            return episode_uuids

        return graphiti_client.run_async(_add_all_episodes())

    def _wait_for_episodes(
        self,
        episode_uuids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600,
    ):
        """Graphiti processes episodes inline during bulk ingest; no Zep-style poll."""
        del timeout
        total = len(episode_uuids)
        if progress_callback:
            progress_callback(
                t("progress.processingComplete", completed=total, total=total),
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
