"""Provider-neutral LLM client."""

import json
import re
from typing import Optional, Dict, Any, List
from openai import AzureOpenAI, OpenAI

from ..config import Config
from .openai_tracing import attach_completion_usage_metadata, wrap_openai_client
from .pipeline_retry import run_pipeline_step
from .vertex_openai import (
    effective_llm_api_key_or_vertex_token,
    effective_llm_base_url,
    is_vertex_ai_enabled,
    vertex_config_present,
)


class LLMClient:
    """Small OpenAI-compatible facade for OpenAI, Azure OpenAI, and Vertex."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.provider = (Config.LLM_PROVIDER or "openai").strip().lower()
        self._vertex = self.provider == "vertex" or is_vertex_ai_enabled()
        self._azure = self.provider == "azure"
        resolved_base = base_url or (
            effective_llm_base_url() if self._vertex else None
        ) or (None if self._azure else Config.LLM_BASE_URL)
        self.base_url = resolved_base
        self.model = model or (
            Config.AZURE_OPENAI_DEPLOYMENT
            if self._azure
            else Config.require_llm_model_name()
        )
        self.api_key = api_key

        if self.provider not in ("openai", "azure", "vertex"):
            raise ValueError("LLM_PROVIDER must be one of: openai, azure, vertex")

        if self._azure:
            if not Config.AZURE_OPENAI_ENDPOINT:
                raise ValueError("AZURE_OPENAI_ENDPOINT is not configured")
            if not (self.api_key or Config.AZURE_OPENAI_API_KEY):
                raise ValueError("AZURE_OPENAI_API_KEY is not configured")
            if not self.model:
                raise ValueError("AZURE_OPENAI_DEPLOYMENT is not configured")
        elif not self._vertex and not (self.api_key or Config.LLM_API_KEY):
            raise ValueError("LLM_API_KEY is not configured")

        if self._vertex and not vertex_config_present():
            raise ValueError(
                "Vertex AI OpenAPI base URL could not be resolved. Check "
                "VERTEX_AI_PROJECT_ID / VERTEX_AI_LOCATION or LLM_BASE_URL."
            )

        self.client: Optional[OpenAI] = None
        if self._azure:
            self.client = wrap_openai_client(
                AzureOpenAI(
                    api_key=self.api_key or Config.AZURE_OPENAI_API_KEY,
                    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                    api_version=Config.AZURE_OPENAI_API_VERSION,
                )
            )
        elif not self._vertex:
            self.client = wrap_openai_client(
                OpenAI(
                    api_key=self.api_key or Config.LLM_API_KEY,
                    base_url=self.base_url,
                )
            )

    def _active_client(self) -> OpenAI:
        if self._vertex:
            key = effective_llm_api_key_or_vertex_token(self.api_key)
            return wrap_openai_client(OpenAI(api_key=key, base_url=self.base_url))
        assert self.client is not None
        return self.client
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """Send a chat completion request and return text content."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        def _complete():
            client = self._active_client()
            return client.chat.completions.create(**kwargs)

        response = run_pipeline_step(
            f"llm_chat_{self.provider}_{self.model}",
            _complete,
            retry_unknown_errors=False,
        )
        attach_completion_usage_metadata(response, model=self.model)
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
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """Send a chat request and parse a JSON object from the response."""
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # Strip Markdown code fences that some providers add despite JSON mode.
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM returned invalid JSON: {cleaned_response}")
