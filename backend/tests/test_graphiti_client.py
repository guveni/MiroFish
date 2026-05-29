from app.config import Config
from app.utils import graphiti_client


def test_vertex_gemini_with_local_embedder_uses_native_gemini_llm(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "vertex")
    monkeypatch.setattr(Config, "LLM_USE_VERTEX_AI", True)
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "google/gemini-3.5-flash")
    monkeypatch.setattr(Config, "VERTEX_AI_PROJECT_ID", "test-project")
    monkeypatch.setattr(Config, "VERTEX_AI_LOCATION", "us-central1")
    monkeypatch.setattr(Config, "GRAPHITI_EMBEDDER", "local")
    monkeypatch.setattr(Config, "GRAPHITI_GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setattr(Config, "GRAPHITI_GEMINI_EMBEDDING_DIM", 1024)
    from graphiti_core.llm_client.gemini_client import GeminiClient

    llm, embedder, _ = graphiti_client._build_local_embedder_clients()

    assert isinstance(llm, GeminiClient)
    assert llm.model == "gemini-3.5-flash"
    assert embedder is not None


def test_ollama_graphiti_clients_use_openai_client(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "qwen2.5:14b-instruct-q4_K_M")
    monkeypatch.setattr(Config, "OLLAMA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(Config, "GRAPHITI_LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setattr(Config, "GRAPHITI_EMBEDDER", "local")
    from graphiti_core.llm_client.openai_client import OpenAIClient

    llm, embedder, _ = graphiti_client._build_local_embedder_clients()

    assert isinstance(llm, OpenAIClient)
    assert llm.model == "qwen2.5:14b-instruct-q4_K_M"
    assert llm.small_model == "qwen2.5:14b-instruct-q4_K_M"
    assert llm.config.base_url == "http://localhost:11434/v1"
    assert embedder is not None


def test_openai_graphiti_clients_default_small_model_to_primary(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(Config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(Config, "LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "gpt-4o-mini")

    llm, _, _ = graphiti_client._build_openai_clients()

    assert llm.model == "gpt-4o-mini"
    assert llm.small_model == "gpt-4o-mini"


def test_gemini_graphiti_clients_use_vertex_and_supported_embedding_default(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "vertex")
    monkeypatch.setattr(Config, "LLM_USE_VERTEX_AI", True)
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "google/gemini-3.5-flash")
    monkeypatch.setattr(Config, "VERTEX_AI_PROJECT_ID", "test-project")
    monkeypatch.setattr(Config, "VERTEX_AI_LOCATION", "us-central1")
    monkeypatch.setattr(Config, "GRAPHITI_GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setattr(Config, "GRAPHITI_GEMINI_EMBEDDING_DIM", 1024)

    llm, embedder, cross_encoder = graphiti_client._build_gemini_clients()

    assert cross_encoder is None
    assert llm.model == "gemini-3.5-flash"
    assert llm.small_model == "gemini-3.5-flash"
    assert llm.client is embedder.client
    assert embedder.client._api_client.vertexai is True
    assert embedder.client._api_client.project == "test-project"
    assert embedder.client._api_client.location == "us-central1"
    assert embedder.config.embedding_model == "gemini-embedding-001"
    assert embedder.config.embedding_dim == 1024
    assert embedder.batch_size == 1


def test_gemini_graphiti_clients_respect_embedding_override(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "vertex")
    monkeypatch.setattr(Config, "LLM_USE_VERTEX_AI", True)
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "gemini-3.5-flash")
    monkeypatch.setattr(Config, "VERTEX_AI_PROJECT_ID", "test-project")
    monkeypatch.setattr(Config, "VERTEX_AI_LOCATION", "global")
    monkeypatch.setattr(Config, "GRAPHITI_GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    monkeypatch.setattr(Config, "GRAPHITI_GEMINI_EMBEDDING_DIM", 1024)

    _, embedder, _ = graphiti_client._build_gemini_clients()

    assert embedder.config.embedding_model == "gemini-embedding-2"
    assert embedder.config.embedding_dim == 1024
    assert embedder.batch_size == 1


def test_gemini_graphiti_clients_require_vertex_location(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "vertex")
    monkeypatch.setattr(Config, "LLM_USE_VERTEX_AI", True)
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "google/gemini-3.5-flash")
    monkeypatch.setattr(Config, "VERTEX_AI_PROJECT_ID", "test-project")
    monkeypatch.setattr(Config, "VERTEX_AI_LOCATION", "")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCP_PROJECT", raising=False)

    try:
        graphiti_client._build_gemini_clients()
    except RuntimeError as exc:
        assert "VERTEX_AI_LOCATION" in str(exc)
    else:
        raise AssertionError("Expected missing Vertex location to fail")


def test_local_embedder_dimension_detection(monkeypatch):
    from app.utils.local_embedder import LocalHuggingFaceEmbedder
    # Mock HuggingFaceEmbeddings to avoid downloading a huge model during pytest
    class MockHuggingFaceEmbeddings:
        def __init__(self, model_name, model_kwargs):
            self.model_name = model_name
            
            # Mock client with get_sentence_embedding_dimension
            class MockClient:
                def get_sentence_embedding_dimension(self):
                    return 384
            self.client = MockClient()

        def embed_query(self, text):
            return [0.1] * 384

    monkeypatch.setattr("langchain_huggingface.HuggingFaceEmbeddings", MockHuggingFaceEmbeddings)

    # Test with default dimension (should detect 384)
    embedder = LocalHuggingFaceEmbedder(model_name="BAAI/bge-small-en-v1.5")
    assert embedder.config.embedding_dim == 384

    # Test with explicitly specified dimension
    embedder_explicit = LocalHuggingFaceEmbedder(model_name="BAAI/bge-small-en-v1.5", embedding_dim=512)
    assert embedder_explicit.config.embedding_dim == 512


def test_graphiti_effective_semaphore_limit_caps_ollama(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(Config, "GRAPHITI_SEMAPHORE_LIMIT", 10)
    monkeypatch.setattr(Config, "GRAPHITI_OLLAMA_CONCURRENCY", 1)
    assert Config.graphiti_effective_semaphore_limit() == 1

    monkeypatch.setattr(Config, "GRAPHITI_OLLAMA_CONCURRENCY", 3)
    assert Config.graphiti_effective_semaphore_limit() == 3


def test_graphiti_effective_semaphore_limit_unchanged_for_cloud(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "vertex")
    monkeypatch.setattr(Config, "GRAPHITI_SEMAPHORE_LIMIT", 10)
    monkeypatch.setattr(Config, "GRAPHITI_OLLAMA_CONCURRENCY", 1)
    assert Config.graphiti_effective_semaphore_limit() == 10


def test_openai_graphiti_client_is_wrapped_when_tracing_enabled(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_some_key")
    monkeypatch.setattr(Config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "gpt-4o-mini")
    monkeypatch.setattr(Config, "GRAPHITI_EMBEDDER", "local")

    from graphiti_core.llm_client.openai_client import OpenAIClient
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key="test")
    orig_create = client.chat.completions.create

    llm, _, _ = graphiti_client._build_local_embedder_clients()

    assert isinstance(llm, OpenAIClient)
    assert llm.client.chat.completions.create != orig_create

