"""
Configurable retries for coarse pipeline steps (LLM / Zep / Vertex GenAI transient failures).

Env (optional):
  PIPELINE_STEP_MAX_RETRIES — extra attempts after the first try (default 2 → 3 tries total).
  PIPELINE_STEP_INITIAL_DELAY_SEC — initial backoff delay (default 1.0).
  PIPELINE_STEP_MAX_DELAY_SEC — cap per wait (default 30.0).
  PIPELINE_STEP_BACKOFF_FACTOR — multiplier (default 2.0).
  PIPELINE_STEP_JITTER — 1/true to randomize waits (default true).
"""

from __future__ import annotations

import random
import socket
import time
from typing import Any, Callable, Iterable, Optional, TypeVar

from ..config import Config
from .logger import get_logger

logger = get_logger("mirofish.pipeline_retry")

T = TypeVar("T")

_TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
_NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}


def _status_code(exc: BaseException) -> Optional[int]:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def is_retryable_pipeline_error(exc: BaseException) -> bool:
    """Return True for transient infrastructure/provider failures only."""
    status = _status_code(exc)
    if status in _NON_RETRYABLE_STATUS_CODES:
        return False
    if status in _TRANSIENT_STATUS_CODES:
        return True
    name = exc.__class__.__name__.lower()
    if "badrequest" in name or "authentication" in name or "permission" in name:
        return False
    if "ratelimit" in name or "timeout" in name or "connection" in name:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, socket.timeout, OSError)):
        return True
    if isinstance(exc, ValueError):
        return False
    return True


def run_pipeline_step(
    step_name: str,
    fn: Callable[[], T],
    *,
    non_retryable_value_messages: Optional[Iterable[str]] = None,
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    backoff_factor: Optional[float] = None,
    jitter: Optional[bool] = None,
    retry_unknown_errors: bool = True,
) -> T:
    """
    Execute fn(); on failure, retry with exponential backoff.

    ``non_retryable_value_messages``: ValueError bodies that must not be retried
    (e.g. Gemini configuration / empty-result contract errors).
    """
    skip_msgs = frozenset(non_retryable_value_messages or ())
    max_r = (
        Config.PIPELINE_STEP_MAX_RETRIES if max_retries is None else max_retries
    )
    delay0 = (
        Config.PIPELINE_STEP_INITIAL_DELAY_SEC
        if initial_delay is None
        else initial_delay
    )
    cap_delay = (
        Config.PIPELINE_STEP_MAX_DELAY_SEC if max_delay is None else max_delay
    )
    bfac = (
        Config.PIPELINE_STEP_BACKOFF_FACTOR
        if backoff_factor is None
        else backoff_factor
    )
    use_jitter = (
        Config.PIPELINE_STEP_JITTER if jitter is None else jitter
    )

    delay = delay0
    last_exc: BaseException | None = None

    for attempt in range(max_r + 1):
        try:
            return fn()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            last_exc = exc
            if isinstance(exc, ValueError) and skip_msgs and str(exc) in skip_msgs:
                raise
            if not is_retryable_pipeline_error(exc):
                logger.warning(
                    "pipeline step %r failed with non-retryable error: %s",
                    step_name,
                    exc,
                )
                raise
            if not retry_unknown_errors and _status_code(exc) is None:
                name = exc.__class__.__name__.lower()
                if not (
                    "ratelimit" in name
                    or "timeout" in name
                    or "connection" in name
                    or isinstance(exc, (TimeoutError, ConnectionError, socket.timeout, OSError))
                ):
                    raise
            if attempt == max_r:
                logger.error(
                    "pipeline step %r failed after %d attempt(s): %s",
                    step_name,
                    max_r + 1,
                    exc,
                )
                raise
            sleep_s = min(delay, cap_delay)
            if use_jitter:
                sleep_s = sleep_s * (0.5 + random.random())
            logger.warning(
                "pipeline step %r attempt %d/%d failed: %s — retry in %.1fs",
                step_name,
                attempt + 1,
                max_r + 1,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)
            delay *= bfac

    assert last_exc is not None
    raise last_exc  # pragma: no cover
