"""
Optional LangSmith tracing for OpenAI-compatible clients (OpenAI SDK).

Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY. See:
https://docs.langchain.com/langsmith/trace-openai
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any, Optional

from openai import AsyncOpenAI, OpenAI

from ..config import Config, _normalize_vertex_genai_model

logger = logging.getLogger(__name__)

_LANGSMITH_OPENAI_PATCHED = False


def _truthy(val: Optional[str]) -> bool:
    if not val:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


def is_langsmith_openai_tracing_enabled() -> bool:
    return _truthy(os.environ.get("LANGSMITH_TRACING")) and bool(
        os.environ.get("LANGSMITH_API_KEY", "").strip()
    )


def _env_float(key: str) -> Optional[float]:
    raw = (os.environ.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_value(key: str) -> Optional[str]:
    raw = (os.environ.get(key) or "").strip()
    return raw or None


def langsmith_ls_provider() -> str:
    """Return the LangSmith provider name for the configured LLM backend."""
    override = _env_value("LANGSMITH_LS_PROVIDER")
    if override:
        return override

    provider = (_env_value("LLM_PROVIDER") or Config.LLM_PROVIDER or "openai").lower()
    if (
        provider == "vertex"
        or _truthy(os.environ.get("LLM_USE_VERTEX_AI"))
        or Config.LLM_USE_VERTEX_AI
    ):
        return "google_vertexai"
    if provider == "azure":
        return "azure"
    return "openai"


def langsmith_ls_model_name(model: Optional[str]) -> Optional[str]:
    """Return the model id LangSmith should use for pricing lookup."""
    override = _env_value("LANGSMITH_LS_MODEL_NAME")
    if override:
        return override
    if not model:
        return None
    if langsmith_ls_provider() == "google_vertexai":
        return _normalize_vertex_genai_model(model)
    return model


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, Mapping):
                return dict(dumped)
        except Exception:
            pass
    attrs = (
        "usage",
        "service_tier",
        "prompt_tokens",
        "completion_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "input_tokens_details",
        "output_tokens_details",
        "audio_tokens",
        "cached_tokens",
        "reasoning_tokens",
    )
    attr_dict = {
        key: getattr(value, key)
        for key in attrs
        if hasattr(value, key) and getattr(value, key) is not None
    }
    if attr_dict:
        return attr_dict
    return {}


def _first_int(data: Mapping[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return int(value)
    return None


def _usage_costs(input_tokens: int, output_tokens: int) -> dict[str, float]:
    in_per_m = _env_float("LLM_INPUT_USD_PER_1M")
    out_per_m = _env_float("LLM_OUTPUT_USD_PER_1M")
    costs: dict[str, float] = {}
    if in_per_m is not None:
        costs["input_cost"] = round(input_tokens / 1_000_000.0 * in_per_m, 8)
    if out_per_m is not None:
        costs["output_cost"] = round(output_tokens / 1_000_000.0 * out_per_m, 8)
    if costs:
        costs["total_cost"] = round(
            costs.get("input_cost", 0.0) + costs.get("output_cost", 0.0), 8
        )
    return costs


def _usage_metadata_from_usage(
    usage: Any,
    *,
    service_tier: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    usage_dict = _as_dict(usage)
    if not usage_dict:
        return None

    input_tokens = _first_int(usage_dict, "prompt_tokens", "input_tokens")
    output_tokens = _first_int(usage_dict, "completion_tokens", "output_tokens")
    if input_tokens is None and output_tokens is None:
        return None

    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    total_tokens = (
        _first_int(usage_dict, "total_tokens") or input_tokens + output_tokens
    )

    recognized_service_tier = (
        service_tier if service_tier in ("priority", "flex") else None
    )
    service_tier_prefix = (
        f"{recognized_service_tier}_" if recognized_service_tier else ""
    )
    prompt_details = _as_dict(
        usage_dict.get("prompt_tokens_details") or usage_dict.get("input_tokens_details")
    )
    completion_details = _as_dict(
        usage_dict.get("completion_tokens_details")
        or usage_dict.get("output_tokens_details")
    )
    input_token_details = {
        "audio": prompt_details.get("audio_tokens"),
        f"{service_tier_prefix}cache_read": prompt_details.get("cached_tokens"),
    }
    output_token_details = {
        "audio": completion_details.get("audio_tokens"),
        f"{service_tier_prefix}reasoning": completion_details.get("reasoning_tokens"),
    }

    if recognized_service_tier:
        input_token_details[recognized_service_tier] = input_tokens - (
            input_token_details.get(f"{service_tier_prefix}cache_read") or 0
        )
        output_token_details[recognized_service_tier] = output_tokens - (
            output_token_details.get(f"{service_tier_prefix}reasoning") or 0
        )

    metadata: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_token_details": {},
        "output_token_details": {},
    }
    clean_input_details = {
        key: value for key, value in input_token_details.items() if value is not None
    }
    clean_output_details = {
        key: value for key, value in output_token_details.items() if value is not None
    }
    if clean_input_details:
        metadata["input_token_details"] = clean_input_details
    if clean_output_details:
        metadata["output_token_details"] = clean_output_details
    metadata.update(_usage_costs(input_tokens, output_tokens))
    return metadata


def build_usage_metadata_from_completion(completion: Any) -> Optional[dict[str, Any]]:
    """Build LangSmith usage_metadata from an OpenAI-compatible completion."""
    outputs = completion
    if hasattr(outputs, "parse") and callable(outputs.parse):
        try:
            outputs = outputs.parse()
        except Exception:
            pass

    output_dict = _as_dict(outputs)
    usage = output_dict.get("usage")
    if usage is None:
        usage = getattr(outputs, "usage", None)
    service_tier = output_dict.get("service_tier") or getattr(
        outputs, "service_tier", None
    )
    return _usage_metadata_from_usage(usage, service_tier=service_tier)


def _with_manual_costs(usage_metadata: Any) -> Any:
    if not isinstance(usage_metadata, Mapping):
        return usage_metadata
    enriched = dict(usage_metadata)
    input_tokens = int(enriched.get("input_tokens") or 0)
    output_tokens = int(enriched.get("output_tokens") or 0)
    enriched.update(_usage_costs(input_tokens, output_tokens))
    return enriched


def _patch_langsmith_openai() -> None:
    """Patch LangSmith OpenAI wrapper hooks once for Vertex pricing metadata."""
    global _LANGSMITH_OPENAI_PATCHED
    if _LANGSMITH_OPENAI_PATCHED:
        return
    try:
        import langsmith.wrappers._openai as ls_openai
    except Exception as e:
        logger.debug("LangSmith OpenAI patch skipped: %s", e)
        return

    original_process = ls_openai._process_chat_completion
    original_infer = ls_openai._infer_invocation_params

    def _process_chat_completion(outputs: Any):
        processed = original_process(outputs)
        if isinstance(processed, dict):
            usage_metadata = processed.get("usage_metadata")
            if usage_metadata:
                processed["usage_metadata"] = _with_manual_costs(usage_metadata)
            else:
                usage_metadata = build_usage_metadata_from_completion(outputs)
                if usage_metadata:
                    processed["usage_metadata"] = usage_metadata
        return processed

    def _infer_invocation_params(
        model_type: str,
        provider: str,
        prepopulated_invocation_params: dict,
        use_responses_api: bool,
        kwargs: dict,
    ):
        params = original_infer(
            model_type,
            provider,
            prepopulated_invocation_params,
            use_responses_api,
            kwargs,
        )
        override_provider = prepopulated_invocation_params.get("ls_provider")
        override_model = prepopulated_invocation_params.get("ls_model_name")
        if override_provider:
            params["ls_provider"] = override_provider
        if override_model:
            params["ls_model_name"] = override_model
        nested = params.get("ls_invocation_params")
        if isinstance(nested, dict):
            nested.pop("ls_provider", None)
            nested.pop("ls_model_name", None)
        return params

    ls_openai._process_chat_completion = _process_chat_completion
    ls_openai._infer_invocation_params = _infer_invocation_params
    _LANGSMITH_OPENAI_PATCHED = True


def langsmith_tracing_metadata(model: Optional[str]) -> dict[str, Any]:
    provider = langsmith_ls_provider()
    model_name = langsmith_ls_model_name(model)
    metadata: dict[str, Any] = {"ls_provider": provider}
    invocation_params: dict[str, str] = {"ls_provider": provider}
    if model_name:
        metadata["ls_model_name"] = model_name
        invocation_params["ls_model_name"] = model_name
    metadata["ls_invocation_params"] = invocation_params
    return metadata


def attach_completion_usage_metadata(
    completion: Any,
    *,
    model: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Deprecated compatibility shim; usage metadata is attached during tracing."""
    logger.debug(
        "attach_completion_usage_metadata is a no-op; LangSmith usage metadata "
        "is added by the OpenAI wrapper output processor."
    )


def wrap_openai_client(
    client: OpenAI | AsyncOpenAI, *, model: Optional[str] = None
) -> OpenAI | AsyncOpenAI:
    """
    Return a LangSmith-instrumented client when tracing is enabled.
    """
    if not is_langsmith_openai_tracing_enabled():
        return client
    try:
        _patch_langsmith_openai()
        from langsmith.wrappers import wrap_openai

        return wrap_openai(
            client,
            tracing_extra={"metadata": langsmith_tracing_metadata(model)},
        )
    except Exception:
        return client


def wrap_camel_model(model_obj: Any, model_name: str) -> Any:
    """Wrap internal OpenAI clients of a camel-ai Model object with LangSmith tracing."""
    if not is_langsmith_openai_tracing_enabled():
        return model_obj
    try:
        if hasattr(model_obj, "_client") and model_obj._client:
            model_obj._client = wrap_openai_client(model_obj._client, model=model_name)
        if hasattr(model_obj, "_async_client") and model_obj._async_client:
            model_obj._async_client = wrap_openai_client(model_obj._async_client, model=model_name)
    except Exception as e:
        logger.warning("Failed to wrap camel model with LangSmith tracing: %s", e)
    return model_obj

