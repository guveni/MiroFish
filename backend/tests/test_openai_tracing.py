from types import SimpleNamespace

from app.config import Config
from app.utils import openai_tracing


def test_langsmith_vertex_provider_and_model_name(monkeypatch):
    monkeypatch.delenv("LANGSMITH_LS_PROVIDER", raising=False)
    monkeypatch.delenv("LANGSMITH_LS_MODEL_NAME", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_USE_VERTEX_AI", raising=False)
    monkeypatch.setattr(Config, "LLM_PROVIDER", "vertex")
    monkeypatch.setattr(Config, "LLM_USE_VERTEX_AI", True)

    assert openai_tracing.langsmith_ls_provider() == "google_vertexai"
    assert (
        openai_tracing.langsmith_ls_model_name("google/gemini-3.5-flash")
        == "gemini-3.5-flash"
    )


def test_langsmith_provider_and_model_env_overrides(monkeypatch):
    monkeypatch.setenv("LANGSMITH_LS_PROVIDER", "custom_provider")
    monkeypatch.setenv("LANGSMITH_LS_MODEL_NAME", "custom-model")
    monkeypatch.setattr(Config, "LLM_PROVIDER", "vertex")
    monkeypatch.setattr(Config, "LLM_USE_VERTEX_AI", True)

    assert openai_tracing.langsmith_ls_provider() == "custom_provider"
    assert (
        openai_tracing.langsmith_ls_model_name("google/gemini-3.5-flash")
        == "custom-model"
    )


def test_build_usage_metadata_from_completion_with_manual_costs(monkeypatch):
    monkeypatch.setenv("LLM_INPUT_USD_PER_1M", "2")
    monkeypatch.setenv("LLM_OUTPUT_USD_PER_1M", "6")
    completion = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )
    )

    usage_metadata = openai_tracing.build_usage_metadata_from_completion(completion)

    assert usage_metadata == {
        "input_tokens": 1000,
        "output_tokens": 500,
        "total_tokens": 1500,
        "input_token_details": {},
        "output_token_details": {},
        "input_cost": 0.002,
        "output_cost": 0.003,
        "total_cost": 0.005,
    }


def test_build_usage_metadata_from_completion_supports_token_details(monkeypatch):
    monkeypatch.delenv("LLM_INPUT_USD_PER_1M", raising=False)
    monkeypatch.delenv("LLM_OUTPUT_USD_PER_1M", raising=False)
    completion = SimpleNamespace(
        service_tier="priority",
        usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2},
        },
    )

    usage_metadata = openai_tracing.build_usage_metadata_from_completion(completion)

    assert usage_metadata == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_token_details": {"priority_cache_read": 3, "priority": 7},
        "output_token_details": {"priority_reasoning": 2, "priority": 3},
    }


def test_tracing_metadata_includes_invocation_overrides(monkeypatch):
    monkeypatch.delenv("LANGSMITH_LS_PROVIDER", raising=False)
    monkeypatch.delenv("LANGSMITH_LS_MODEL_NAME", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(Config, "LLM_PROVIDER", "vertex")
    monkeypatch.setattr(Config, "LLM_USE_VERTEX_AI", True)

    metadata = openai_tracing.langsmith_tracing_metadata("google/gemini-3.5-flash")

    assert metadata["ls_provider"] == "google_vertexai"
    assert metadata["ls_model_name"] == "gemini-3.5-flash"
    assert metadata["ls_invocation_params"] == {
        "ls_provider": "google_vertexai",
        "ls_model_name": "gemini-3.5-flash",
    }


def test_langsmith_infer_patch_promotes_invocation_overrides():
    openai_tracing._patch_langsmith_openai()

    import langsmith.wrappers._openai as ls_openai

    params = ls_openai._infer_invocation_params(
        "chat",
        "openai",
        {"ls_provider": "google_vertexai", "ls_model_name": "gemini-3.5-flash"},
        False,
        {"model": "google/gemini-3.5-flash", "temperature": 0.1},
    )

    assert params["ls_provider"] == "google_vertexai"
    assert params["ls_model_name"] == "gemini-3.5-flash"
    assert "ls_provider" not in params["ls_invocation_params"]
    assert "ls_model_name" not in params["ls_invocation_params"]


def test_langsmith_output_processor_patch_adds_manual_costs(monkeypatch):
    monkeypatch.setenv("LLM_INPUT_USD_PER_1M", "1")
    monkeypatch.setenv("LLM_OUTPUT_USD_PER_1M", "2")
    openai_tracing._patch_langsmith_openai()

    import langsmith.wrappers._openai as ls_openai

    class Completion:
        def model_dump(self):
            return {
                "id": "completion-id",
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "total_tokens": 1500,
                },
            }

    processed = ls_openai._process_chat_completion(Completion())

    assert processed["usage_metadata"]["input_cost"] == 0.001
    assert processed["usage_metadata"]["output_cost"] == 0.001
    assert processed["usage_metadata"]["total_cost"] == 0.002
