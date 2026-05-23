"""Graphiti paging helpers with Zep-compatible return shapes."""

from __future__ import annotations

from typing import Any, List

from . import graphiti_client


def fetch_all_nodes(group_id: str) -> List[Any]:
    async def _fetch():
        from graphiti_core.nodes import EntityNode

        return await EntityNode.get_by_group_ids(
            graphiti_client.get_client().driver,
            [group_id],
        )

    try:
        return graphiti_client.run_async(_fetch())
    except Exception:
        return []


def fetch_all_edges(group_id: str) -> List[Any]:
    async def _fetch():
        from graphiti_core.edges import EntityEdge

        return await EntityEdge.get_by_group_ids(
            graphiti_client.get_client().driver,
            [group_id],
        )

    try:
        return graphiti_client.run_async(_fetch())
    except Exception:
        return []
