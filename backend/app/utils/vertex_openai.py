"""
Vertex AI Gemini via the OpenAI-compatible Chat Completions endpoint.

Uses Application Default Credentials (adc); no static API key.
See: https://cloud.google.com/vertex-ai/generative-ai/docs/migrate/openai/overview
"""

import os
from typing import Optional, Tuple
from urllib.parse import urlparse


_VERTEX_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)
_vertex_credentials = None


def _truthy(env_val: Optional[str]) -> bool:
    if not env_val:
        return False
    return env_val.strip().lower() in ("1", "true", "yes", "on")


def is_vertex_ai_enabled() -> bool:
    return (
        (os.environ.get("LLM_PROVIDER") or "").strip().lower() == "vertex"
        or _truthy(os.environ.get("LLM_USE_VERTEX_AI"))
    )


def _vertex_project_id() -> Optional[str]:
    return (
        os.environ.get("VERTEX_AI_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT")
    )


def _vertex_location() -> str:
    """Unset when synthesizing openapi URL yields no URL (explicit LLM_BASE_URL may omit this)."""
    return (os.environ.get("VERTEX_AI_LOCATION") or "").strip()


def _looks_like_vertex_openapi_endpoint(url: str) -> bool:
    """True if LLM_BASE_URL points at Vertex Chat Completions (OpenAPI) roots."""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path or ""
    except ValueError:
        return False
    return "aiplatform.googleapis.com" in host and "/endpoints/openapi" in path


def vertex_openapi_base_url() -> Optional[str]:
    """
    Effective OpenAI base URL for Vertex (…/endpoints/openapi), without trailing slash.

    Uses LLM_BASE_URL only when it looks like a regional Vertex openapi endpoint,
    otherwise ignores leftover values (e.g. DashScope URLs from copying .env.example)
    unless LLM_VERTEX_USE_EXPLICIT_BASE_URL=true.
    """
    explicit = (os.environ.get("LLM_BASE_URL") or "").strip()
    if explicit:
        force = _truthy(os.environ.get("LLM_VERTEX_USE_EXPLICIT_BASE_URL"))
        if force or _looks_like_vertex_openapi_endpoint(explicit):
            return explicit.rstrip("/")
    project = _vertex_project_id()
    location = _vertex_location()
    if not project:
        return None
    if not location:
        return None
    api_version = (os.environ.get("VERTEX_AI_OPENAI_API_VERSION") or "v1").strip().lstrip("/")
    if api_version not in ("v1", "v1beta1"):
        api_version = "v1"
    loc = location.lower()
    if loc == "global":
        return (
            f"https://aiplatform.googleapis.com/{api_version}/projects/"
            f"{project}/locations/global/endpoints/openapi"
        )
    return (
        f"https://{location}-aiplatform.googleapis.com/{api_version}/projects/"
        f"{project}/locations/{location}/endpoints/openapi"
    )


def vertex_config_present() -> bool:
    """True if Vertex mode can resolve an endpoint URL."""
    return vertex_openapi_base_url() is not None


def get_vertex_access_token() -> str:
    """OAuth2 access token (refreshes when stale)."""
    global _vertex_credentials
    from google.auth.transport.requests import Request
    from google.auth import default

    if _vertex_credentials is None:
        _vertex_credentials, _ = default(scopes=list(_VERTEX_SCOPES))
    if not _vertex_credentials.valid:
        _vertex_credentials.refresh(Request())
    if not _vertex_credentials.token:
        raise RuntimeError("Vertex ADC refresh did not yield an access token")
    return _vertex_credentials.token


def effective_llm_base_url() -> Optional[str]:
    """LLM_BASE_URL equivalent: Vertex-built URL when enabled, else env LLM_BASE_URL."""
    if is_vertex_ai_enabled():
        return vertex_openapi_base_url()
    raw = os.environ.get("LLM_BASE_URL")
    return raw.strip().rstrip("/") if raw else None


def effective_llm_api_key_or_vertex_token(static_key: Optional[str] = None) -> str:
    """
    API key/token for OpenAI SDK. Uses Vertex OAuth token when Vertex is enabled,
    otherwise the provided static key or LLM_API_KEY.
    """
    if is_vertex_ai_enabled():
        return get_vertex_access_token()
    key = static_key if static_key is not None else os.environ.get("LLM_API_KEY", "")
    return key or ""


def prepare_camel_openai_env() -> Tuple[str, str, str]:
    """
    Populate OPENAI_API_KEY / OPENAI_API_BASE_URL for camel-ai ModelFactory.
    Call before ModelFactory.create().

    Returns:
        (llm_model_name, llm_base_url_display, hint for logs)
    """
    llm_model = os.environ.get("LLM_MODEL_NAME", "").strip()
    if not llm_model:
        raise ValueError("LLM_MODEL_NAME must be explicitly configured in .env")

    base = effective_llm_base_url()
    token_or_key = effective_llm_api_key_or_vertex_token()

    if not token_or_key:
        raise ValueError(
            "Missing LLM credentials: set LLM_API_KEY, or LLM_USE_VERTEX_AI=true "
            "with ADC (gcloud auth application-default login) and project/region or LLM_BASE_URL."
        )

    os.environ["OPENAI_API_KEY"] = token_or_key
    display = ""
    if base:
        os.environ["OPENAI_API_BASE_URL"] = base
        display = base[:60] + ("…" if len(base) > 60 else "")
    else:
        os.environ.pop("OPENAI_API_BASE_URL", None)
        display = "default"

    hint = "vertex-adc" if is_vertex_ai_enabled() else "api-key"
    return llm_model, display, hint
