"""Local HuggingFace embedder for Graphiti."""

import asyncio
from typing import Iterable

from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig


class LocalHuggingFaceEmbedder(EmbedderClient):
    def __init__(self, model_name: str, embedding_dim: int = 1024):
        from langchain_huggingface import HuggingFaceEmbeddings

        self.config = EmbedderConfig(embedding_dim=embedding_dim)
        self._model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
        )

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        # Handle string input
        text = str(input_data)
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._model.embed_query, text)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._model.embed_documents, input_data_list)
