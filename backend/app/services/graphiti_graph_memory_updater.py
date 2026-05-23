"""Graphiti-backed graph memory updater."""

from __future__ import annotations

import threading
import time
from queue import Queue
from typing import Any, Dict, List, Optional

from .zep_graph_memory_updater import AgentActivity, ZepGraphMemoryUpdater, logger
from ..utils import graphiti_client


class GraphitiGraphMemoryUpdater(ZepGraphMemoryUpdater):
    def __init__(self, graph_id: str, api_key: Optional[str] = None):
        self.graph_id = graph_id
        self.client = graphiti_client.get_client()
        self._activity_queue: Queue = Queue()
        self._platform_buffers: Dict[str, List[AgentActivity]] = {
            "twitter": [],
            "reddit": [],
        }
        self._buffer_lock = threading.Lock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._total_activities = 0
        self._total_sent = 0
        self._total_items_sent = 0
        self._failed_count = 0
        self._skipped_count = 0
        logger.info(
            "GraphitiGraphMemoryUpdater initialized: graph_id=%s, batch_size=%s",
            graph_id,
            self.BATCH_SIZE,
        )

    def _send_batch_activities(self, activities: List[AgentActivity], platform: str):
        if not activities:
            return

        for attempt in range(self.MAX_RETRIES):
            try:
                async def _add():
                    from graphiti_core.nodes import EpisodeType
                    from graphiti_core.utils.bulk_utils import RawEpisode

                    bulk_episodes = [
                        RawEpisode(
                            name=f"{platform}-activity-{int(time.time())}-{i}",
                            content=activity.to_episode_text(),
                            source=EpisodeType.text,
                            source_description=f"MiroFish {platform} simulation activity",
                            reference_time=graphiti_client.utcnow(),
                        )
                        for i, activity in enumerate(activities)
                    ]

                    return await self.client.add_episode_bulk(
                        bulk_episodes=bulk_episodes,
                        group_id=self.graph_id,
                        **graphiti_client.get_ontology(self.graph_id),
                    )

                graphiti_client.run_async(_add())
                self._total_sent += 1
                self._total_items_sent += len(activities)
                display_name = self._get_platform_display_name(platform)
                logger.info(
                    "Sent %d %s activities to Graphiti graph %s",
                    len(activities),
                    display_name,
                    self.graph_id,
                )
                return
            except Exception as exc:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(
                        "Graphiti batch send failed (attempt %d/%d): %s",
                        attempt + 1,
                        self.MAX_RETRIES,
                        exc,
                    )
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error("Graphiti batch send failed after retries: %s", exc)
                    self._failed_count += 1
