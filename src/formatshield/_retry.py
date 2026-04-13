"""
FormatShield retry utilities — exponential backoff for HTTP backend calls.

Provides a simple, dependency-minimal retry mechanism built on :mod:`anyio`
that is compatible with both ``asyncio`` and ``trio`` event loops.

The retry logic implements truncated exponential backoff with optional jitter,
matching the approach used by the ``instructor`` library for robust API calls.

# Pattern inspired by: instructor retry patterns, guidance error handling

Example::

    from formatshield._retry import RetryConfig, with_retry

    cfg = RetryConfig(max_attempts=3, base_delay=1.0)

    async def _call() -> str:
        return await my_http_client.post(...)

    result = await with_retry(_call, cfg, retryable=(RateLimitError,))
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TypeVar

import anyio

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    """Immutable configuration for the retry behaviour of a backend.

    Parameters
    ----------
    max_attempts:
        Maximum total number of attempts (including the first).  ``1`` means
        no retries.  Default ``3``.
    base_delay:
        Initial back-off delay in seconds.  Doubles on each subsequent attempt.
        Default ``1.0``.
    max_delay:
        Upper cap on the back-off delay in seconds.  Default ``30.0``.
    jitter:
        When ``True`` (default) apply ±50% uniform jitter to each delay to
        avoid thundering-herd effects across concurrent requests.
    retryable_exceptions:
        Tuple of exception types that should trigger a retry.  Defaults to
        ``(Exception,)`` (retry on any error).  Pass a narrower set (e.g.
        ``(RateLimitError, TimeoutError)``) for production use.

    Example::

        cfg = RetryConfig(max_attempts=4, base_delay=0.5, max_delay=10.0)
        assert cfg.max_attempts == 4
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = field(default_factory=lambda: (Exception,))

    def delay_for(self, attempt: int) -> float:
        """Return the back-off delay in seconds for *attempt* (0-indexed).

        Parameters
        ----------
        attempt:
            Zero-indexed attempt number.  Attempt 0 was the first try; delay
            is computed for the upcoming retry.

        Returns
        -------
        float
            Delay in seconds, capped at :attr:`max_delay`.
        """
        delay = min(self.base_delay * (2**attempt), self.max_delay)
        if self.jitter:
            delay *= random.uniform(0.5, 1.5)  # noqa: S311 — not cryptographic
        return delay


#: Sensible default configuration — 3 attempts with 1 s initial delay.
DEFAULT_RETRY = RetryConfig()

#: Aggressive retry for rate-limited API calls — 5 attempts, longer delays.
API_RETRY = RetryConfig(
    max_attempts=5,
    base_delay=2.0,
    max_delay=60.0,
    jitter=True,
)


async def with_retry(
    coro_fn: Callable[[], Coroutine[object, object, T]],
    config: RetryConfig = DEFAULT_RETRY,
    *,
    retryable: tuple[type[Exception], ...] | None = None,
    operation_name: str = "operation",
) -> T:
    """Call *coro_fn* and retry on failure according to *config*.

    Parameters
    ----------
    coro_fn:
        A zero-argument async callable that performs the operation.  It must
        return a coroutine that yields the result on success or raises on
        failure.
    config:
        :class:`RetryConfig` controlling the retry behaviour.
    retryable:
        Override the retryable exception types for this specific call.  When
        ``None``, falls back to ``config.retryable_exceptions``.
    operation_name:
        Human-readable label used in log messages.

    Returns
    -------
    T
        The value returned by *coro_fn* on success.

    Raises
    ------
    Exception
        Re-raises the last exception if all attempts fail.

    Example::

        import groq

        async def _call() -> str:
            return await client.chat.completions.create(...)

        result = await with_retry(
            _call,
            RetryConfig(max_attempts=3),
            retryable=(groq.RateLimitError, groq.InternalServerError),
            operation_name="groq.generate",
        )
    """
    effective_retryable: tuple[type[Exception], ...] = (
        retryable if retryable is not None else config.retryable_exceptions
    )

    last_exc: BaseException | None = None

    for attempt in range(config.max_attempts):
        try:
            return await coro_fn()
        except BaseException as exc:
            # Only retry on the configured retryable exception types.
            if not isinstance(exc, effective_retryable):
                raise

            last_exc = exc
            remaining = config.max_attempts - attempt - 1

            if remaining == 0:
                logger.warning(
                    "%s: all %d attempt(s) exhausted — raising last error: %s",
                    operation_name,
                    config.max_attempts,
                    exc,
                )
                raise

            delay = config.delay_for(attempt)
            logger.warning(
                "%s: attempt %d/%d failed (%s: %s). Retrying in %.2f s …",
                operation_name,
                attempt + 1,
                config.max_attempts,
                type(exc).__name__,
                exc,
                delay,
            )
            await anyio.sleep(delay)

    # Should never reach here but makes the type checker happy.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{operation_name}: retry loop exited without result")  # pragma: no cover
