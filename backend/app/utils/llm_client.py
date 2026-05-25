"""Provider-neutral LLM client."""

import json
import logging
import re
from typing import Optional, Dict, Any, List
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from ..config import Config
from .openai_tracing import wrap_openai_client
from .pipeline_retry import run_pipeline_step, run_pipeline_step_async

logger = logging.getLogger(__name__)
from .vertex_openai import (
    effective_llm_api_key_or_vertex_token,
    effective_llm_base_url,
    is_vertex_ai_enabled,
    vertex_config_present,
)


def _strip_markdown_json_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    return cleaned.strip()


def _close_truncated_json(text: str) -> str:
    """Close an incomplete JSON object, array, or string."""
    in_string = False
    escape = False
    stack: list[str] = []

    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == '{':
            stack.append('{')
        elif ch == '[':
            stack.append('[')
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()

    result = text.rstrip()
    if in_string:
        result += '"'
    for opener in reversed(stack):
        result += ']' if opener == '[' else '}'
    return result


def _repair_root_object_trailing_garbage(text: str) -> Optional[str]:
    """Truncate junk appended after the last complete root-level field value."""
    brace_depth = 0
    bracket_depth = 0
    in_string = False
    escape = False
    last_root_value_end = -1
    n = len(text)

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
                if brace_depth == 1 and bracket_depth == 0:
                    j = i + 1
                    while j < n and text[j] in ' \t\r\n':
                        j += 1
                    if j >= n:
                        last_root_value_end = i + 1
                    elif text[j] not in ',}:':
                        last_root_value_end = i + 1
            continue
        if ch == '"':
            in_string = True
        elif ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
        elif ch == '[':
            bracket_depth += 1
        elif ch == ']':
            bracket_depth -= 1

    if last_root_value_end < 0:
        return None

    candidate = text[:last_root_value_end].rstrip().rstrip(',')
    return f"{candidate}\n}}"


def parse_llm_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON from an LLM response, repairing common formatting defects."""
    cleaned = _strip_markdown_json_fences(text)
    attempts = (
        cleaned,
        _close_truncated_json(cleaned),
        _repair_root_object_trailing_garbage(cleaned),
    )
    seen: set[str] = set()
    for candidate in attempts:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
            if candidate != cleaned:
                logger.warning(
                    "Repaired malformed LLM JSON (original length=%d, repaired length=%d)",
                    len(cleaned),
                    len(candidate),
                )
            return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError(f"LLM returned invalid JSON: {cleaned}")


class LLMClient:
    """Small OpenAI-compatible facade for OpenAI, Azure OpenAI, Vertex, and Ollama."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.provider = (Config.LLM_PROVIDER or "openai").strip().lower()
        self._vertex = self.provider == "vertex" or is_vertex_ai_enabled()
        self._azure = self.provider == "azure"
        self._ollama = self.provider == "ollama"

        if self._ollama:
            resolved_base = base_url or Config.OLLAMA_BASE_URL
        elif self._vertex:
            resolved_base = base_url or effective_llm_base_url()
        elif self._azure:
            resolved_base = None
        else:
            resolved_base = base_url or Config.LLM_BASE_URL
        self.base_url = resolved_base

        self.model = model or (
            Config.AZURE_OPENAI_DEPLOYMENT
            if self._azure
            else Config.require_llm_model_name()
        )
        self.api_key = api_key

        if self.provider not in ("openai", "azure", "vertex", "ollama"):
            raise ValueError("LLM_PROVIDER must be one of: openai, azure, vertex, ollama")

        if self._azure:
            if not Config.AZURE_OPENAI_ENDPOINT:
                raise ValueError("AZURE_OPENAI_ENDPOINT is not configured")
            if not (self.api_key or Config.AZURE_OPENAI_API_KEY):
                raise ValueError("AZURE_OPENAI_API_KEY is not configured")
            if not self.model:
                raise ValueError("AZURE_OPENAI_DEPLOYMENT is not configured")
        elif self._ollama:
            pass  # Ollama needs no API key; dummy value used below.
        elif not self._vertex and not (self.api_key or Config.LLM_API_KEY):
            raise ValueError("LLM_API_KEY is not configured")

        if self._vertex and not vertex_config_present():
            raise ValueError(
                "Vertex AI OpenAPI base URL could not be resolved. Check "
                "VERTEX_AI_PROJECT_ID / VERTEX_AI_LOCATION or LLM_BASE_URL."
            )

        self.client: Optional[OpenAI] = None
        self.async_client: Optional[AsyncOpenAI] = None
        if self._azure:
            self.client = wrap_openai_client(
                AzureOpenAI(
                    api_key=self.api_key or Config.AZURE_OPENAI_API_KEY,
                    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                    api_version=Config.AZURE_OPENAI_API_VERSION,
                ),
                model=self.model,
            )
            self.async_client = AsyncAzureOpenAI(
                api_key=self.api_key or Config.AZURE_OPENAI_API_KEY,
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                api_version=Config.AZURE_OPENAI_API_VERSION,
            )
        elif self._ollama or not self._vertex:
            key = self.api_key or ("ollama" if self._ollama else Config.LLM_API_KEY)
            self.client = wrap_openai_client(
                OpenAI(api_key=key, base_url=self.base_url),
                model=self.model,
            )
            self.async_client = AsyncOpenAI(
                api_key=key, base_url=self.base_url,
            )

    def _active_client(self) -> OpenAI:
        if self._vertex:
            key = effective_llm_api_key_or_vertex_token(self.api_key)
            return wrap_openai_client(
                OpenAI(api_key=key, base_url=self.base_url),
                model=self.model,
            )
        assert self.client is not None
        return self.client

    def _active_async_client(self) -> AsyncOpenAI:
        if self._vertex:
            key = effective_llm_api_key_or_vertex_token(self.api_key)
            return AsyncOpenAI(api_key=key, base_url=self.base_url)
        assert self.async_client is not None
        return self.async_client
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None
    ) -> str:
        """Send a chat completion request and return text content."""
        import time
        actual_max_tokens = max_tokens if max_tokens is not None else Config.LLM_CHAT_MAX_TOKENS
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": actual_max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        prompt_len = sum(len(m.get("content", "")) for m in messages)
        logger.info(f"[LLM] Sending request to {self.provider}:{self.model} (prompt length ~{prompt_len} chars, response_format={response_format})")
        start_time = time.time()
        
        def _complete():
            client = self._active_client()
            return client.chat.completions.create(**kwargs)

        try:
            response = run_pipeline_step(
                f"llm_chat_{self.provider}_{self.model}",
                _complete,
                retry_unknown_errors=False,
            )
            elapsed = time.time() - start_time
            logger.info(f"[LLM] Received response from {self.provider}:{self.model} in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[LLM] Request failed on {self.provider}:{self.model} in {elapsed:.2f}s: {e}")
            raise

        if not response.choices:
            raise ValueError("LLM returned no choices")
        choice = response.choices[0]
        message = choice.message
        if message is None:
            finish = getattr(choice, "finish_reason", None) or "unknown"
            raise ValueError(
                f"LLM returned no message content (finish_reason={finish!r}). "
                "Increase max_tokens or reduce the requested output size."
            )
        content = message.content or ""
        # Some models include hidden reasoning tags in content; remove them.
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    async def achat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None
    ) -> str:
        """Async chat completion request returning text content."""
        import time
        actual_max_tokens = max_tokens if max_tokens is not None else Config.LLM_CHAT_MAX_TOKENS
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": actual_max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        prompt_len = sum(len(m.get("content", "")) for m in messages)
        logger.info(f"[LLM] Sending async request to {self.provider}:{self.model} (prompt length ~{prompt_len} chars, response_format={response_format})")
        start_time = time.time()

        async def _complete():
            client = self._active_async_client()
            return await client.chat.completions.create(**kwargs)

        try:
            response = await run_pipeline_step_async(
                f"llm_chat_{self.provider}_{self.model}",
                _complete,
                retry_unknown_errors=False,
            )
            elapsed = time.time() - start_time
            logger.info(f"[LLM] Received async response from {self.provider}:{self.model} in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[LLM] Async request failed on {self.provider}:{self.model} in {elapsed:.2f}s: {e}")
            raise

        if not response.choices:
            raise ValueError("LLM returned no choices")
        choice = response.choices[0]
        message = choice.message
        if message is None:
            finish = getattr(choice, "finish_reason", None) or "unknown"
            raise ValueError(
                f"LLM returned no message content (finish_reason={finish!r}). "
                "Increase max_tokens or reduce the requested output size."
            )
        content = message.content or ""
        return re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Send a chat request and parse a JSON object from the response."""
        actual_max_tokens = max_tokens if max_tokens is not None else Config.LLM_JSON_MAX_TOKENS
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=actual_max_tokens,
            response_format={"type": "json_object"}
        )
        return parse_llm_json_response(response)

    async def achat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Async chat request parsed as a JSON object."""
        actual_max_tokens = max_tokens if max_tokens is not None else Config.LLM_JSON_MAX_TOKENS
        response = await self.achat(
            messages=messages,
            temperature=temperature,
            max_tokens=actual_max_tokens,
            response_format={"type": "json_object"}
        )
        return parse_llm_json_response(response)
