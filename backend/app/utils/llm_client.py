"""Provider-neutral LLM client powered by LangChain."""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from ..config import Config
from .pipeline_retry import run_pipeline_step, run_pipeline_step_async

logger = logging.getLogger(__name__)


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
                if last_root_value_end < 0 and brace_depth == 1 and bracket_depth == 0:
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
            if brace_depth == 0 and bracket_depth == 0 and last_root_value_end < 0:
                return text[:i+1]
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
    """LangChain-powered facade for OpenAI, Azure OpenAI, Vertex, Ollama, and Lambda."""
    
    @classmethod
    def for_composer(cls) -> "LLMClient":
        """Factory method to build a composer-specific LLMClient, or fallback to the primary one."""
        if not Config.REPORT_USE_COMPOSER or not Config.COMPOSER_LLM_PROVIDER:
            return cls()
        return cls(
            provider=Config.COMPOSER_LLM_PROVIDER,
            api_key=Config.COMPOSER_LLM_API_KEY or None,
            base_url=Config.COMPOSER_LLM_BASE_URL or None,
            model=Config.COMPOSER_LLM_MODEL_NAME or None,
        )

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None
    ):
        self.provider = (provider or Config.LLM_PROVIDER or "openai").strip().lower()
        self._vertex = self.provider == "vertex"
        self._azure = self.provider == "azure"
        self._ollama = self.provider == "ollama"
        self._lambda = self.provider == "lambda"

        self.model = model or (Config.AZURE_OPENAI_DEPLOYMENT if self._azure else Config.require_llm_model_name())
        self.api_key = api_key

        if self.provider not in ("openai", "azure", "vertex", "ollama", "lambda"):
            raise ValueError(f"LLM provider {self.provider} must be one of: openai, azure, vertex, ollama, lambda")

        # Resolve base URL
        if self._ollama:
            self.base_url = base_url or Config.OLLAMA_BASE_URL
        elif self._lambda:
            self.base_url = base_url or Config.LAMBDA_BASE_URL
        else:
            self.base_url = base_url or Config.LLM_BASE_URL

        # Instantiate the correct LangChain Chat Model
        if self._azure:
            endpoint = base_url or (Config.COMPOSER_LLM_BASE_URL if hasattr(Config, "COMPOSER_LLM_BASE_URL") and Config.COMPOSER_LLM_BASE_URL else Config.AZURE_OPENAI_ENDPOINT)
            key = api_key or (Config.COMPOSER_LLM_API_KEY if hasattr(Config, "COMPOSER_LLM_API_KEY") and Config.COMPOSER_LLM_API_KEY else Config.AZURE_OPENAI_API_KEY)
            deployment = self.model
            
            if not endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT is not configured")
            if not key:
                raise ValueError("AZURE_OPENAI_API_KEY is not configured")
            if not deployment:
                raise ValueError("AZURE_OPENAI_DEPLOYMENT is not configured")

            self._model = AzureChatOpenAI(
                openai_api_key=key,
                azure_endpoint=endpoint,
                api_version=Config.AZURE_OPENAI_API_VERSION,
                azure_deployment=deployment,
            )
        elif self._ollama:
            self._model = ChatOpenAI(
                openai_api_key="ollama",
                base_url=self.base_url,
                model_name=self.model,
            )
        elif self._vertex:
            from .vertex_openai import effective_llm_base_url, effective_llm_api_key_or_vertex_token
            base_url = effective_llm_base_url()
            api_key = effective_llm_api_key_or_vertex_token(self.api_key)
            self._model = ChatOpenAI(
                openai_api_key=api_key,
                base_url=base_url,
                model_name=self.model,
            )
        else:
            # OpenAI or Lambda (which is OpenAI compatible)
            key = self.api_key or (
                Config.COMPOSER_LLM_API_KEY
                if hasattr(Config, "COMPOSER_LLM_API_KEY") and Config.COMPOSER_LLM_API_KEY and provider
                else (Config.LAMBDA_API_KEY if self._lambda else Config.LLM_API_KEY)
            )

            if self.provider == "openai" and not key:
                raise ValueError("LLM_API_KEY or COMPOSER_LLM_API_KEY is not configured")
            if self.provider == "lambda" and not key:
                raise ValueError("LAMBDA_API_KEY is not configured")
                
            self._model = ChatOpenAI(
                openai_api_key=key,
                base_url=self.base_url,
                model_name=self.model,
            )

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Any]:
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        return lc_messages

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None
    ) -> str:
        """Send a chat completion request and return text content using LangChain."""
        actual_max_tokens = max_tokens if max_tokens is not None else Config.LLM_CHAT_MAX_TOKENS
        
        # Check global LLM Cache
        from .llm_cache import LLMCache
        cache = LLMCache.get_instance()
        params = {
            "temperature": temperature,
            "max_tokens": actual_max_tokens,
            "response_format": response_format,
            "provider": self.provider,
            "model": self.model,
        }
        cached_response = cache.get(messages, params)
        if cached_response is not None:
            return cached_response

        lc_messages = self._convert_messages(messages)
        
        # Build bound model with dynamic invocation parameters
        kwargs = {
            "temperature": temperature,
            "max_tokens": actual_max_tokens,
        }
        if response_format:
            if self.provider in ("openai", "vertex", "azure"):
                kwargs["response_format"] = response_format
            else:
                kwargs["model_kwargs"] = {"response_format": response_format}
            
        bound_model = self._model.bind(**kwargs)
        
        prompt_len = sum(len(m.get("content", "")) for m in messages)
        logger.info(f"[LLM] Sending request via LangChain to {self.provider}:{self.model} (prompt length ~{prompt_len} chars)")
        start_time = time.time()
        
        def _complete():
            return bound_model.invoke(lc_messages)

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

        content = response.content or ""
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        
        # Store in Cache
        cache.set(messages, params, content)
        return content

    async def achat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None
    ) -> str:
        """Async chat completion request returning text content using LangChain."""
        actual_max_tokens = max_tokens if max_tokens is not None else Config.LLM_CHAT_MAX_TOKENS

        # Check global LLM Cache
        from .llm_cache import LLMCache
        cache = LLMCache.get_instance()
        params = {
            "temperature": temperature,
            "max_tokens": actual_max_tokens,
            "response_format": response_format,
            "provider": self.provider,
            "model": self.model,
        }
        cached_response = cache.get(messages, params)
        if cached_response is not None:
            return cached_response

        lc_messages = self._convert_messages(messages)
        
        kwargs = {
            "temperature": temperature,
            "max_tokens": actual_max_tokens,
        }
        if response_format:
            if self.provider in ("openai", "vertex", "azure"):
                kwargs["response_format"] = response_format
            else:
                kwargs["model_kwargs"] = {"response_format": response_format}
            
        bound_model = self._model.bind(**kwargs)

        prompt_len = sum(len(m.get("content", "")) for m in messages)
        logger.info(f"[LLM] Sending async request via LangChain to {self.provider}:{self.model} (prompt length ~{prompt_len} chars)")
        start_time = time.time()

        async def _complete():
            return await bound_model.ainvoke(lc_messages)

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

        content = response.content or ""
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        
        # Store in Cache
        cache.set(messages, params, content)
        return content
    
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
