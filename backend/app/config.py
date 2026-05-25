"""
Configuration management.
Loads settings from the project root .env file.
"""

import os
from dotenv import load_dotenv

# Load the project root .env file.
# Path: MiroFish/.env (relative to backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # Fall back to process environment in production deployments.
    load_dotenv(override=True)


def _normalize_vertex_genai_model(model: str | None) -> str:
    """Normalize OpenAI-compatible Vertex model ids for google-genai."""
    m = (model or '').strip()
    if m.startswith('google/'):
        return m.split('/', 1)[1].strip()
    return m


def _implicit_gemini_web_search_model(llm_model: str, vertex_enabled: bool) -> str:
    """
    Reuse LLM_MODEL_NAME only when it is clearly a Vertex Gemini model.

    This avoids sending provider-specific chat models such as qwen-plus to the
    Vertex GenAI grounding endpoint when GEMINI_WEB_SEARCH_MODEL is omitted.
    """
    normalized = _normalize_vertex_genai_model(llm_model)
    if not normalized:
        return ''
    if (llm_model or '').strip().startswith('google/'):
        return normalized
    if vertex_enabled and normalized.lower().startswith('gemini-'):
        return normalized
    return ''


def _llm_provider() -> str:
    provider = (os.environ.get('LLM_PROVIDER') or '').strip().lower()
    if provider:
        return provider
    if os.environ.get('LLM_USE_VERTEX_AI', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        return 'vertex'
    return 'openai'


class Config:
    """Flask configuration."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # JSON settings. Keep Unicode readable instead of escaping it as \uXXXX.
    JSON_AS_ASCII = False
    
    # LLM settings. LLM_PROVIDER choices: openai, azure, vertex.
    LLM_PROVIDER = _llm_provider()
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = (os.environ.get('LLM_MODEL_NAME') or '').strip()
    LLM_USE_VERTEX_AI = (
        LLM_PROVIDER == 'vertex'
        or os.environ.get('LLM_USE_VERTEX_AI', '').strip().lower() in ('1', 'true', 'yes', 'on')
    )
    AZURE_OPENAI_ENDPOINT = (os.environ.get('AZURE_OPENAI_ENDPOINT') or '').strip()
    AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
    AZURE_OPENAI_API_VERSION = (
        os.environ.get('AZURE_OPENAI_API_VERSION') or '2024-02-15-preview'
    ).strip()
    AZURE_OPENAI_DEPLOYMENT = (
        os.environ.get('AZURE_OPENAI_DEPLOYMENT') or LLM_MODEL_NAME
    ).strip()
    VERTEX_AI_PROJECT_ID = os.environ.get('VERTEX_AI_PROJECT_ID', '')
    VERTEX_AI_LOCATION = (os.environ.get('VERTEX_AI_LOCATION') or '').strip()
    # Large structured JSON (ontology, profiles). Gemini 3.x on Vertex can exhaust
    # smaller budgets via internal reasoning before emitting visible content.
    LLM_JSON_MAX_TOKENS = int(os.environ.get('LLM_JSON_MAX_TOKENS', '8192'))
    LLM_CHAT_MAX_TOKENS = int(os.environ.get('LLM_CHAT_MAX_TOKENS', '8192'))

    # Gemini (Vertex) web search grounding via Google Search, not Discovery Engine.
    # Requires VERTEX_AI_PROJECT_ID or GOOGLE_CLOUD_PROJECT plus VERTEX_AI_LOCATION; uses ADC.
    # When GEMINI_WEB_SEARCH_MODEL is unset, reuse LLM_MODEL_NAME only for Vertex Gemini models.
    # OpenAI-compatible names like google/foo are normalized for Vertex GenAI.
    GEMINI_WEB_SEARCH_MODEL = (
        _normalize_vertex_genai_model(os.environ.get('GEMINI_WEB_SEARCH_MODEL'))
        or _implicit_gemini_web_search_model(LLM_MODEL_NAME, LLM_USE_VERTEX_AI)
    )
    GEMINI_WEB_SEARCH_MAX_QUERIES = int(os.environ.get('GEMINI_WEB_SEARCH_MAX_QUERIES', '5'))
    GEMINI_WEB_SEARCH_MAX_CHARS = int(os.environ.get('GEMINI_WEB_SEARCH_MAX_CHARS', '25000'))
    GEMINI_WEB_SEARCH_MAX_OUTPUT_TOKENS = int(os.environ.get('GEMINI_WEB_SEARCH_MAX_OUTPUT_TOKENS', '8192'))
    
    # Graph memory backend. Unset/empty defaults to Neo4j + Graphiti.
    GRAPH_BACKEND = (os.environ.get('GRAPH_BACKEND') or 'neo4j').strip().lower()
    NEO4J_URI = (os.environ.get('NEO4J_URI') or 'bolt://localhost:7687').strip()
    NEO4J_USER = (os.environ.get('NEO4J_USER') or 'neo4j').strip()
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD') or 'mirofish-dev'
    NEO4J_DATABASE = (os.environ.get('NEO4J_DATABASE') or 'neo4j').strip()
    GRAPHITI_EMBEDDER = (os.environ.get('GRAPHITI_EMBEDDER') or 'auto').strip().lower()
    GRAPHITI_RERANKER = (os.environ.get('GRAPHITI_RERANKER') or 'auto').strip().lower()
    GRAPHITI_SEMAPHORE_LIMIT = int(os.environ.get('GRAPHITI_SEMAPHORE_LIMIT', '10'))
    GRAPHITI_SEARCH_TIMEOUT = int(os.environ.get('GRAPHITI_SEARCH_TIMEOUT', '30'))
    # Expected seconds per Graphiti bulk batch; used for in-batch progress heartbeat only.
    GRAPHITI_BATCH_HEARTBEAT_SECONDS = int(os.environ.get('GRAPHITI_BATCH_HEARTBEAT_SECONDS', '90'))
    GRAPHITI_BATCH_SIZE = int(os.environ.get('GRAPHITI_BATCH_SIZE', '10'))
    GRAPHITI_LOCAL_EMBEDDING_MODEL = (
        os.environ.get('GRAPHITI_LOCAL_EMBEDDING_MODEL') or 'Alibaba-NLP/gte-large-en-v1.5'
    )
    GRAPHITI_GEMINI_EMBEDDING_MODEL = (
        os.environ.get('GRAPHITI_GEMINI_EMBEDDING_MODEL') or 'gemini-embedding-001'
    ).strip()
    GRAPHITI_GEMINI_EMBEDDING_DIM = int(os.environ.get('GRAPHITI_GEMINI_EMBEDDING_DIM', '1024'))

    # Legacy Zep settings
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')
    
    # File upload settings
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # Text processing settings
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50
    
    # OASIS simulation settings
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    SIM_CONFIG_MAX_WORKERS = int(os.environ.get('SIM_CONFIG_MAX_WORKERS', '8'))
    
    # OASIS platform actions
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report Agent settings
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))

    # Pipeline step retries: one initial attempt plus PIPELINE_STEP_MAX_RETRIES retries.
    PIPELINE_STEP_MAX_RETRIES = int(os.environ.get('PIPELINE_STEP_MAX_RETRIES', '2'))
    PIPELINE_STEP_INITIAL_DELAY_SEC = float(
        os.environ.get('PIPELINE_STEP_INITIAL_DELAY_SEC', '1.0')
    )
    PIPELINE_STEP_MAX_DELAY_SEC = float(
        os.environ.get('PIPELINE_STEP_MAX_DELAY_SEC', '30.0')
    )
    PIPELINE_STEP_BACKOFF_FACTOR = float(
        os.environ.get('PIPELINE_STEP_BACKOFF_FACTOR', '2.0')
    )
    PIPELINE_STEP_JITTER = os.environ.get('PIPELINE_STEP_JITTER', 'true').strip().lower() in (
        '1', 'true', 'yes', 'on',
    )
    RESUME_FROM_CHECKPOINT = os.environ.get(
        'RESUME_FROM_CHECKPOINT', 'true'
    ).strip().lower() in ('1', 'true', 'yes', 'on')

    @classmethod
    def require_llm_model_name(cls) -> str:
        if not cls.LLM_MODEL_NAME:
            raise ValueError("LLM_MODEL_NAME must be explicitly configured in .env")
        return cls.LLM_MODEL_NAME

    @classmethod
    def normalized_graph_backend(cls) -> str:
        backend = (cls.GRAPH_BACKEND or 'neo4j').strip().lower()
        if backend in ('neo4j', 'graphiti'):
            return 'graphiti'
        if backend == 'zep':
            return 'zep'
        return backend

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        errors = []
        if cls.LLM_PROVIDER not in ('openai', 'azure', 'vertex'):
            errors.append("LLM_PROVIDER must be one of: openai, azure, vertex")
        if cls.LLM_PROVIDER == 'azure':
            if not cls.AZURE_OPENAI_ENDPOINT:
                errors.append("AZURE_OPENAI_ENDPOINT is not configured")
            if not cls.AZURE_OPENAI_API_KEY:
                errors.append("AZURE_OPENAI_API_KEY is not configured")
            if not cls.AZURE_OPENAI_DEPLOYMENT:
                errors.append("AZURE_OPENAI_DEPLOYMENT is not configured")
        elif cls.LLM_USE_VERTEX_AI:
            from .utils.vertex_openai import vertex_config_present

            if not vertex_config_present():
                errors.append(
                    "Vertex AI requires VERTEX_AI_PROJECT_ID or GOOGLE_CLOUD_PROJECT "
                    "plus VERTEX_AI_LOCATION, or a complete Vertex OpenAPI LLM_BASE_URL "
                    "with LLM_VERTEX_USE_EXPLICIT_BASE_URL=true when needed."
                )
        elif not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY is not configured")
        graph_backend = cls.normalized_graph_backend()
        if graph_backend == 'zep':
            if not cls.ZEP_API_KEY:
                errors.append("ZEP_API_KEY is not configured")
        elif graph_backend == 'graphiti':
            if not cls.NEO4J_URI:
                errors.append("NEO4J_URI is not configured")
            if not cls.NEO4J_PASSWORD:
                errors.append("NEO4J_PASSWORD is not configured")
        else:
            errors.append("GRAPH_BACKEND must be one of: neo4j, graphiti, zep")
        if not cls.LLM_MODEL_NAME and cls.LLM_PROVIDER != 'azure':
            errors.append("LLM_MODEL_NAME is not configured")
        return errors
