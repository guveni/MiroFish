"""Graph backend selector."""

from ..config import Config

_GRAPHITI_ALIASES = frozenset({"neo4j", "graphiti"})


def active_backend() -> str:
    return (Config.GRAPH_BACKEND or "neo4j").strip().lower()


def is_zep() -> bool:
    return active_backend() == "zep"


def is_graphiti() -> bool:
    return active_backend() in _GRAPHITI_ALIASES


def is_neo4j() -> bool:
    return is_graphiti()


def is_available() -> tuple[bool, str | None]:
    if is_zep():
        return bool(Config.ZEP_API_KEY), None if Config.ZEP_API_KEY else "api.zepApiKeyMissing"
    if is_graphiti():
        from ..utils import graphiti_client

        return graphiti_client.is_available()
    return False, "api.graphBackendUnavailable"
