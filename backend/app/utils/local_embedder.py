"""Local HuggingFace embedder for Graphiti."""

import asyncio
import os
from typing import Iterable

from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig


class LocalHuggingFaceEmbedder(EmbedderClient):
    def __init__(self, model_name: str, embedding_dim: int | None = None):
        from langchain_huggingface import HuggingFaceEmbeddings

        device = os.environ.get("HF_EMBEDDER_DEVICE", "cpu")
        self._model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device, "trust_remote_code": True},
        )

        # Dynamically detect embedding dimension if not explicitly provided
        if embedding_dim is None:
            try:
                # Try HuggingFaceEmbeddings/SentenceTransformer's method first
                if hasattr(self._model, "client") and hasattr(self._model.client, "get_sentence_embedding_dimension"):
                    detected_dim = self._model.client.get_sentence_embedding_dimension()
                else:
                    # Fallback to embedding a dummy string
                    detected_dim = len(self._model.embed_query("dim_test"))
                embedding_dim = detected_dim
            except Exception:
                # Fallback default values
                embedding_dim = 384 if "bge-small" in model_name else 1024

        self.config = EmbedderConfig(embedding_dim=embedding_dim)

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        loop = asyncio.get_running_loop()

        if isinstance(input_data, str):
            return await loop.run_in_executor(None, self._model.embed_query, input_data)

        if isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            vecs = await loop.run_in_executor(None, self._model.embed_documents, input_data)
            return vecs[0] if len(input_data) == 1 else vecs

        # Token-ID input (Iterable[int] / Iterable[Iterable[int]]): not supported
        # by HuggingFaceEmbeddings; fall back to stringifying.
        text = " ".join(str(t) for t in input_data) if hasattr(input_data, "__iter__") else str(input_data)
        return await loop.run_in_executor(None, self._model.embed_query, text)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._model.embed_documents, input_data_list)
