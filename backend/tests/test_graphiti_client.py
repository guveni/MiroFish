from app.config import Config
from app.utils import graphiti_client


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
