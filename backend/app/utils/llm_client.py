"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config
from .openai_tracing import wrap_openai_client
from .vertex_openai import (
    effective_llm_api_key_or_vertex_token,
    effective_llm_base_url,
    is_vertex_ai_enabled,
    vertex_config_present,
)


class LLMClient:
    """LLM客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self._vertex = is_vertex_ai_enabled()
        resolved_base = base_url or (
            effective_llm_base_url() if self._vertex else None
        ) or Config.LLM_BASE_URL
        self.base_url = resolved_base
        self.model = model or Config.LLM_MODEL_NAME
        self.api_key = api_key

        if not self._vertex and not (self.api_key or Config.LLM_API_KEY):
            raise ValueError("LLM_API_KEY 未配置")

        if self._vertex and not vertex_config_present():
            raise ValueError(
                "Vertex AI：无法解析 openapi 基础 URL（检查 VERTEX_AI_PROJECT_ID / "
                "VERTEX_AI_LOCATION 或 LLM_BASE_URL）"
            )

        self.client: Optional[OpenAI] = None
        if not self._vertex:
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
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        response = self._active_client().chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            解析后的JSON对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response}")

