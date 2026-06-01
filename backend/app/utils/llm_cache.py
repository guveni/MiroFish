"""
Global caching for LLMClient queries.
Supports exact and simple semantic similarity matching, utilizing Redis with a local SQLite/memory fallback.
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LLMCache:
    """
    Global LLM Cache with Redis and local SQLite fallback.
    Implements exact matching (via hash of prompt + params) and 
    a fast Jaccard-token semantic similarity match for prompt observations.
    """
    _instance: Optional["LLMCache"] = None

    @classmethod
    def get_instance(cls) -> "LLMCache":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Configure cache parameters from environment/config
        self.enabled = os.environ.get("LLM_CACHE_ENABLED", "True").lower() in ("true", "1", "yes", "on")
        self.provider = os.environ.get("LLM_CACHE_PROVIDER", "redis").lower()
        self.redis_url = os.environ.get("REDIS_URL") or os.environ.get("LLM_CACHE_REDIS_URL") or "redis://localhost:6379/0"
        self.semantic_threshold = float(os.environ.get("LLM_CACHE_SEMANTIC_THRESHOLD", "0.95"))
        
        self.redis_client = None
        self.sqlite_conn = None
        
        if not self.enabled:
            logger.info("LLM Caching is globally disabled.")
            return

        # Initialize the configured provider
        if self.provider == "redis":
            try:
                import redis
                logger.info(f"Connecting LLM Cache to Redis at {self.redis_url}...")
                self.redis_client = redis.Redis.from_url(self.redis_url, socket_timeout=2.0)
                # Test connection
                self.redis_client.ping()
                logger.info("LLM Cache: Redis connection successful.")
            except Exception as e:
                logger.warning(f"LLM Cache: Redis connection failed ({e}). Falling back to local SQLite cache.")
                self.provider = "sqlite"

        if self.provider == "sqlite" or self.provider == "memory":
            try:
                # Use standard file-based SQLite or in-memory SQLite depending on provider
                db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../uploads"))
                os.makedirs(db_dir, exist_ok=True)
                db_path = ":memory:" if self.provider == "memory" else os.path.join(db_dir, "llm_cache.db")
                
                logger.info(f"LLM Cache: Initializing SQLite persistent cache at {db_path}...")
                self.sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
                self._init_sqlite_db()
            except Exception as e:
                logger.error(f"LLM Cache: SQLite initialization failed ({e}). Disabling cache.")
                self.enabled = False

    def _init_sqlite_db(self):
        cursor = self.sqlite_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY,
                prompt_text TEXT,
                response_text TEXT,
                metadata TEXT,
                created_at REAL
            )
        """)
        # Index for semantic lookups if needed
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)")
        self.sqlite_conn.commit()

    def _normalize_and_serialize_prompt(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> Tuple[str, str]:
        """Normalize messages to generate a stable exact match cache key and flat prompt text."""
        # Convert messages to a canonical JSON string for exact hashing
        canonical_messages = []
        flat_text_parts = []
        for msg in messages:
            role = msg.get("role", "user").strip().lower()
            content = msg.get("content", "").strip()
            canonical_messages.append({"role": role, "content": content})
            flat_text_parts.append(f"{role}: {content}")

        # Combine messages structure with temperature, response_format, etc.
        serialized_params = json.dumps(params, sort_keys=True)
        serialized_messages = json.dumps(canonical_messages, sort_keys=True)
        
        # Exact match key: hash of messages + invocation params
        key_raw = f"{serialized_messages}||{serialized_params}"
        cache_key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
        
        return cache_key, "\n".join(flat_text_parts)

    def _tokenize(self, text: str) -> set:
        """Helper to tokenize prompt strings for fast Jaccard similarity checks."""
        # Lowercase, keep words/numbers
        words = re.findall(r"\w+", text.lower())
        return set(words)

    def _compute_jaccard_similarity(self, tokens1: set, tokens2: set) -> float:
        """Compute the Jaccard similarity coefficient between two sets of tokens."""
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        return intersection / union

    def get(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> Optional[str]:
        """Retrieve a cached response if available, checking exact match first, then semantic match."""
        if not self.enabled:
            return None

        cache_key, prompt_text = self._normalize_and_serialize_prompt(messages, params)

        # 1. Exact match lookup
        if self.provider == "redis" and self.redis_client:
            try:
                cached_data = self.redis_client.get(f"llm_cache:exact:{cache_key}")
                if cached_data:
                    logger.info("LLM Cache HIT (Redis Exact)")
                    return cached_data.decode("utf-8")
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")

        elif self.sqlite_conn:
            try:
                cursor = self.sqlite_conn.cursor()
                cursor.execute("SELECT response_text FROM cache_entries WHERE cache_key = ?", (cache_key,))
                row = cursor.fetchone()
                if row:
                    logger.info("LLM Cache HIT (SQLite Exact)")
                    return row[0]
            except Exception as e:
                logger.warning(f"SQLite get failed: {e}")

        # 2. Semantic lookup fallback
        # To scale, we token-match the prompt text (focusing on the last query/observation)
        # against recently cached observations.
        logger.debug("LLM Cache MISS (Exact), checking semantic match...")
        
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        
        if not last_user_msg:
            # Fall back to whole prompt
            last_user_msg = prompt_text

        query_tokens = self._tokenize(last_user_msg)
        if not query_tokens:
            return None

        best_match_response = None
        best_similarity = 0.0

        if self.provider == "redis" and self.redis_client:
            try:
                # In Redis, we retrieve the set of semantic-eligible entries.
                # To keep it extremely efficient, we scan recent prompt observational entries.
                # A production system can use Redisearch or redis vector indices;
                # for our 100% stable out-of-the-box solution, we fetch a sliding window of recent observations.
                keys = self.redis_client.keys("llm_cache:semantic:*")
                # To avoid overloading Redis, we only check up to 100 recent entries.
                for key in keys[:100]:
                    metadata_bytes = self.redis_client.get(key)
                    if metadata_bytes:
                        meta = json.loads(metadata_bytes.decode("utf-8"))
                        cached_last_msg = meta.get("last_user_msg", "")
                        cached_tokens = self._tokenize(cached_last_msg)
                        sim = self._compute_jaccard_similarity(query_tokens, cached_tokens)
                        if sim > best_similarity:
                            best_similarity = sim
                            best_match_response = meta.get("response_text")
            except Exception as e:
                logger.warning(f"Redis semantic scan failed: {e}")

        elif self.sqlite_conn:
            try:
                cursor = self.sqlite_conn.cursor()
                # Fetch up to 100 recent cache entries to search semantically
                cursor.execute("SELECT prompt_text, response_text, metadata FROM cache_entries ORDER BY created_at DESC LIMIT 100")
                rows = cursor.fetchall()
                for prompt, response, metadata_str in rows:
                    last_msg = ""
                    if metadata_str:
                        meta = json.loads(metadata_str)
                        last_msg = meta.get("last_user_msg", "")
                    if not last_msg:
                        # Fallback to prompt extraction
                        last_msg = prompt
                    
                    cached_tokens = self._tokenize(last_msg)
                    sim = self._compute_jaccard_similarity(query_tokens, cached_tokens)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_match_response = response
            except Exception as e:
                logger.warning(f"SQLite semantic scan failed: {e}")

        if best_similarity >= self.semantic_threshold and best_match_response:
            logger.info(f"LLM Cache HIT (Semantic Similarity: {best_similarity:.2f} >= {self.semantic_threshold})")
            return best_match_response

        return None

    def set(self, messages: List[Dict[str, str]], params: Dict[str, Any], response_text: str):
        """Cache a newly generated response."""
        if not self.enabled or not response_text:
            return

        cache_key, prompt_text = self._normalize_and_serialize_prompt(messages, params)
        
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        if not last_user_msg:
            last_user_msg = prompt_text

        meta = {
            "last_user_msg": last_user_msg,
            "response_text": response_text,
            "params": params,
            "created_at": time.time()
        }
        meta_str = json.dumps(meta)

        # 1. Cache in Redis
        if self.provider == "redis" and self.redis_client:
            try:
                # Exact key (ttl: 7 days)
                self.redis_client.setex(f"llm_cache:exact:{cache_key}", 604800, response_text)
                # Semantic index entry (ttl: 7 days)
                self.redis_client.setex(f"llm_cache:semantic:{cache_key}", 604800, meta_str)
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")

        # 2. Cache in SQLite
        elif self.sqlite_conn:
            try:
                cursor = self.sqlite_conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO cache_entries (cache_key, prompt_text, response_text, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                    (cache_key, prompt_text, response_text, meta_str, time.time())
                )
                self.sqlite_conn.commit()
            except Exception as e:
                logger.warning(f"SQLite set failed: {e}")
