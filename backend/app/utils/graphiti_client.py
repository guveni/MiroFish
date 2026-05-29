"""Shared Graphiti client and sync bridge."""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from ..config import Config, is_vertex_gemini_chat_model
from .logger import get_logger
from .vertex_openai import effective_llm_api_key_or_vertex_token, effective_llm_base_url

logger = get_logger("mirofish.graphiti")

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()
_client: Any | None = None
_client_lock = threading.Lock()
_indices_ready = False
_ontology_lock = threading.Lock()
_ontology_cache: Dict[str, Dict[str, Any]] = {}


def _vertex_project_id() -> str:
    return (
        (Config.VERTEX_AI_PROJECT_ID or "").strip()
        or (os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip()
        or (os.environ.get("GCP_PROJECT") or "").strip()
    )


def _vertex_location() -> str:
    return (Config.VERTEX_AI_LOCATION or "").strip()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop, _loop_thread
    with _loop_lock:
        if _loop and _loop.is_running():
            return _loop

        loop = asyncio.new_event_loop()

        def run_loop() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=run_loop, name="GraphitiLoop", daemon=True)
        thread.start()
        _loop = loop
        _loop_thread = thread
        return loop


def run_async(coro, timeout: float | None = None):
    """Run a Graphiti coroutine on the shared background event loop.

    Args:
        coro: Awaitable to schedule.
        timeout: Max seconds to wait. None means wait forever.
    """
    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


def _graphiti_small_model(primary_model: str) -> str:
    """Model for Graphiti ModelSize.small steps (edge dedup, etc.).

    We intentionally keep Graphiti "small" steps on the same model as the primary
    Graphiti LLM to avoid introducing a separate provider/model setting.
    """
    return primary_model


def _graphiti_llm_config(
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    from graphiti_core.llm_client.config import LLMConfig

    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        small_model=_graphiti_small_model(model),
    )


def _create_graphiti_openai_llm(config: Any) -> Any:
    from graphiti_core.llm_client.openai_client import OpenAIClient
    from openai import AsyncOpenAI
    from app.utils.openai_tracing import wrap_openai_client

    raw_client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
    wrapped_client = wrap_openai_client(raw_client, model=config.model)
    return OpenAIClient(config=config, client=wrapped_client)


def _set_default_openai_env() -> None:
    """Provide Graphiti's OpenAI-compatible defaults from MiroFish config."""
    if Config.LLM_PROVIDER == "azure":
        return
    if Config.LLM_PROVIDER == "vertex":
        os.environ.setdefault("OPENAI_API_KEY", effective_llm_api_key_or_vertex_token())
        os.environ.setdefault("OPENAI_BASE_URL", effective_llm_base_url())
    elif Config.LLM_PROVIDER == "lambda":
        # Lambda is OpenAI-compatible; expose credentials/URL via OPENAI_* for
        # Graphiti internals that instantiate OpenAI clients from env vars.
        api_key = (Config.LAMBDA_API_KEY or Config.LLM_API_KEY or "").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = Config.LAMBDA_BASE_URL
        if Config.LLM_MODEL_NAME:
            os.environ["MODEL_NAME"] = Config.LLM_MODEL_NAME
        return
    elif Config.LLM_PROVIDER == "ollama":
        # Force local Ollama so stale cloud OPENAI_* vars cannot leak in.
        os.environ["OPENAI_API_KEY"] = "ollama"
        os.environ["OPENAI_BASE_URL"] = Config.OLLAMA_BASE_URL
        if Config.LLM_MODEL_NAME:
            os.environ["MODEL_NAME"] = Config.LLM_MODEL_NAME
        return
    else:
        if Config.LLM_API_KEY:
            os.environ.setdefault("OPENAI_API_KEY", Config.LLM_API_KEY)
        if Config.LLM_BASE_URL:
            os.environ.setdefault("OPENAI_BASE_URL", Config.LLM_BASE_URL)
    if Config.LLM_MODEL_NAME:
        os.environ.setdefault("MODEL_NAME", Config.LLM_MODEL_NAME)


def _build_openai_clients() -> tuple[Any | None, Any | None, Any | None]:
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

    api_key, base_url, model = _resolve_openai_compat_llm_params()
    llm = _create_graphiti_openai_llm(
        _graphiti_llm_config(api_key=api_key, base_url=base_url, model=model)
    )
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=api_key,
            base_url=base_url,
            embedding_model=os.environ.get("GRAPHITI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
    )
    return llm, embedder, None


def _build_azure_clients() -> tuple[Any | None, Any | None, Any | None]:
    try:
        from openai import AsyncAzureOpenAI
        from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient
        from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
    except Exception as exc:
        raise RuntimeError("Graphiti Azure clients are not available") from exc

    deployment = Config.AZURE_OPENAI_DEPLOYMENT
    from typing import cast
    from app.utils.openai_tracing import wrap_openai_client

    azure_client = cast(
        AsyncAzureOpenAI,
        wrap_openai_client(
            AsyncAzureOpenAI(
                api_key=Config.AZURE_OPENAI_API_KEY,
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                api_version=Config.AZURE_OPENAI_API_VERSION,
            ),
            model=deployment,
        ),
    )
    llm = AzureOpenAILLMClient(
        azure_client=azure_client,
        config=_graphiti_llm_config(model=deployment),
    )
    embedder = AzureOpenAIEmbedderClient(
        azure_client=azure_client,
        model=os.environ.get("GRAPHITI_AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
    )
    return llm, embedder, None


def _build_gemini_clients() -> tuple[Any | None, Any | None, Any | None]:
    try:
        from google import genai
        from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
        from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
    except Exception as exc:
        raise RuntimeError("Graphiti Gemini clients are not available") from exc

    model = Config.require_llm_model_name()
    model = model.split("/", 1)[1] if model.startswith("google/") else model
    client_kwargs: dict[str, Any] = {}
    if Config.LLM_PROVIDER == "vertex" or Config.LLM_USE_VERTEX_AI:
        project = _vertex_project_id()
        location = _vertex_location()
        if not project or not location:
            raise RuntimeError(
                "Graphiti Gemini clients require VERTEX_AI_PROJECT_ID/GOOGLE_CLOUD_PROJECT "
                "and VERTEX_AI_LOCATION when Vertex mode is enabled."
            )
        client_kwargs = {"vertexai": True, "project": project, "location": location}
    else:
        api_key = (
            os.environ.get("GRAPHITI_GEMINI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or Config.LLM_API_KEY
        )
        if api_key:
            client_kwargs = {"api_key": api_key}

    genai_client = genai.Client(**client_kwargs)
    llm = GeminiClient(config=_graphiti_llm_config(model=model), client=genai_client)
    embedder = GeminiEmbedder(
        config=GeminiEmbedderConfig(
            embedding_model=Config.GRAPHITI_GEMINI_EMBEDDING_MODEL,
            embedding_dim=Config.GRAPHITI_GEMINI_EMBEDDING_DIM,
        ),
        client=genai_client,
        batch_size=1,
    )
    return llm, embedder, None


def _resolve_openai_compat_llm_params() -> tuple[str, str | None, str]:
    """Return (api_key, base_url, model) for any OpenAI-compatible LLM provider."""
    if Config.LLM_PROVIDER == "vertex":
        api_key = effective_llm_api_key_or_vertex_token()
        base_url = effective_llm_base_url()
    elif Config.LLM_PROVIDER == "lambda":
        api_key = (Config.LAMBDA_API_KEY or Config.LLM_API_KEY or "").strip()
        base_url = Config.LAMBDA_BASE_URL
    elif Config.LLM_PROVIDER == "ollama":
        api_key = "ollama"
        base_url = Config.OLLAMA_BASE_URL
    else:
        api_key = Config.LLM_API_KEY or ""
        base_url = Config.LLM_BASE_URL
    return api_key, base_url, Config.require_llm_model_name()


def _build_local_embedder_clients() -> tuple[Any, Any, None]:
    """Local HF embedder + LLM (Vertex Gemini native or OpenAI-compatible)."""
    from .local_embedder import LocalHuggingFaceEmbedder

    local_emb = LocalHuggingFaceEmbedder(model_name=Config.GRAPHITI_LOCAL_EMBEDDING_MODEL)

    if Config.LLM_PROVIDER == "vertex" and is_vertex_gemini_chat_model(Config.LLM_MODEL_NAME):
        try:
            llm, _, _ = _build_gemini_clients()
            logger.info(
                "Graphiti LLM: Vertex Gemini native client with local embedder (model=%s)",
                Config.LLM_MODEL_NAME,
            )
            return llm, local_emb, None
        except Exception as exc:
            logger.warning(
                "Vertex Gemini native Graphiti client unavailable, "
                "falling back to OpenAI-compatible: %s",
                exc,
            )

    api_key, base_url, llm_model = _resolve_openai_compat_llm_params()
    llm = _create_graphiti_openai_llm(
        _graphiti_llm_config(api_key=api_key, base_url=base_url, model=llm_model)
    )
    return llm, local_emb, None


def _build_clients() -> tuple[Any | None, Any | None, Any | None]:
    embedder = Config.GRAPHITI_EMBEDDER
    provider = Config.LLM_PROVIDER

    if provider == "azure" or embedder == "azure":
        return _build_azure_clients()

    # Honour explicit local embedder before provider-specific builders so that
    # e.g. LLM_PROVIDER=vertex + GRAPHITI_EMBEDDER=local uses the local HF
    # embedder instead of GeminiEmbedder.
    if embedder == "local":
        return _build_local_embedder_clients()

    # Ollama is always OpenAI-compatible; skip the Gemini path entirely.
    if provider == "ollama":
        return _build_openai_clients()

    if (provider == "vertex" or embedder == "vertex") and is_vertex_gemini_chat_model(
        Config.LLM_MODEL_NAME
    ):
        try:
            return _build_gemini_clients()
        except Exception as exc:
            logger.warning("Falling back to OpenAI-compatible Graphiti clients: %s", exc)

    return _build_openai_clients()


async def _ensure_indices(graphiti: Any) -> None:
    global _indices_ready
    if _indices_ready:
        return
    await graphiti.build_indices_and_constraints()
    _indices_ready = True


def get_client():
    global _client
    with _client_lock:
        if _client is not None:
            return _client

        from graphiti_core import Graphiti
        from graphiti_core.driver.neo4j_driver import Neo4jDriver

        _set_default_openai_env()
        llm_client, embedder, cross_encoder = _build_clients()
        driver = Neo4jDriver(
            Config.NEO4J_URI,
            Config.NEO4J_USER,
            Config.NEO4J_PASSWORD,
            database=Config.NEO4J_DATABASE,
        )
        _client = Graphiti(
            graph_driver=driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
            max_coroutines=Config.graphiti_effective_semaphore_limit(),
        )
    run_async(_ensure_indices(_client))
    return _client


def is_available() -> tuple[bool, str | None]:
    if not Config.NEO4J_URI or not Config.NEO4J_PASSWORD:
        return False, "api.graphBackendUnavailable"
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USER or "", Config.NEO4J_PASSWORD or ""),
        )
        try:
            driver.verify_connectivity()
        finally:
            driver.close()
        return True, None
    except Exception as exc:
        logger.warning("Graphiti backend unavailable: %s", exc)
        return False, "api.graphBackendUnavailable"


def _safe_class_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else " " for ch in name).strip()
    pascal = "".join(part[:1].upper() + part[1:] for part in cleaned.split())
    return pascal or "Entity"


def _safe_attr_name(attr_name: str) -> str:
    reserved = {"uuid", "name", "group_id", "name_embedding", "summary", "created_at"}
    normalized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in attr_name).strip("_")
    if not normalized:
        normalized = "value"
    if normalized.lower() in reserved:
        return f"entity_{normalized}"
    return normalized


def _model_from_definition(name: str, description: str, attributes: list[dict[str, Any]]):
    annotations: dict[str, Any] = {}
    fields: dict[str, Any] = {"__doc__": description}
    for attr_def in attributes:
        attr_name = _safe_attr_name(str(attr_def.get("name") or "value"))
        attr_desc = str(attr_def.get("description") or attr_name)
        annotations[attr_name] = Optional[str]
        fields[attr_name] = Field(default=None, description=attr_desc)
    fields["__annotations__"] = annotations
    return type(_safe_class_name(name), (BaseModel,), fields)


def register_ontology(group_id: str, ontology: dict[str, Any]) -> dict[str, Any]:
    with _ontology_lock:
        if group_id in _ontology_cache:
            return _ontology_cache[group_id]

        entity_types = {}
        for entity_def in ontology.get("entity_types", []):
            name = str(entity_def.get("name") or "Entity")
            description = str(entity_def.get("description") or f"A {name} entity.")
            entity_types[name] = _model_from_definition(
                name,
                description,
                entity_def.get("attributes", []),
            )

        edge_types = {}
        edge_type_map: dict[tuple[str, str], list[str]] = {}
        for edge_def in ontology.get("edge_types", []):
            name = str(edge_def.get("name") or "RELATES_TO")
            description = str(edge_def.get("description") or f"A {name} relationship.")
            edge_types[name] = _model_from_definition(
                name,
                description,
                edge_def.get("attributes", []),
            )
            for source_target in edge_def.get("source_targets", []):
                source = str(source_target.get("source") or "Entity")
                target = str(source_target.get("target") or "Entity")
                edge_type_map.setdefault((source, target), []).append(name)

        if edge_types and not edge_type_map:
            edge_type_map[("Entity", "Entity")] = list(edge_types.keys())

        cached = {
            "entity_types": entity_types or None,
            "edge_types": edge_types or None,
            "edge_type_map": edge_type_map or None,
        }
        _ontology_cache[group_id] = cached
        return cached


def get_ontology(group_id: str) -> dict[str, Any]:
    with _ontology_lock:
        return _ontology_cache.get(
            group_id,
            {"entity_types": None, "edge_types": None, "edge_type_map": None},
        )


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
