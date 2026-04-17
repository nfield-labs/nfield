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
from typing import Any, NamedTuple, TypeVar

import anyio

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Reask / retry failure types
# ---------------------------------------------------------------------------


class FailedAttempt(NamedTuple):
    """Record of a single failed generation attempt.

    Attributes
    ----------
    attempt_number:
        1-indexed attempt number (1 = first try, 2 = first retry, …).
    exception:
        The exception raised by this attempt.
    raw_output:
        The raw output string from this attempt (may be invalid JSON or empty).
    reask_prompt:
        The modified prompt sent on this attempt (includes error feedback when
        reask is enabled; equals the original prompt on attempt 1).

    Example::

        fa = FailedAttempt(attempt_number=1, exception=ValueError("bad"), raw_output="")
        assert fa.attempt_number == 1
    """

    attempt_number: int
    exception: BaseException
    raw_output: str
    reask_prompt: str = ""


class FormatShieldRetryException(Exception):  # noqa: N818
    """Raised when all retry attempts have been exhausted.

    Carries the complete history of failed attempts so callers can inspect
    what went wrong on each try.

    Args:
        message: Human-readable description of the failure.
        failed_attempts: Ordered list of :class:`FailedAttempt` records.

    Example::

        exc = FormatShieldRetryException(
            "All 3 attempts failed",
            failed_attempts=[FailedAttempt(1, ValueError("bad"), "")],
        )
        assert len(exc.failed_attempts) == 1
    """

    def __init__(
        self,
        message: str,
        failed_attempts: list[FailedAttempt] | None = None,
    ) -> None:
        super().__init__(message)
        self.failed_attempts: list[FailedAttempt] = failed_attempts or []

    @property
    def last_attempt(self) -> FailedAttempt | None:
        """Return the most recent :class:`FailedAttempt`, or ``None`` if empty."""
        return self.failed_attempts[-1] if self.failed_attempts else None

    @property
    def total_token_usage(self) -> int:
        """Approximate total tokens across all attempts.

        Returns the sum of ``len(attempt.raw_output)`` as a rough proxy for
        token consumption when exact token counts are unavailable.

        Returns:
            Integer sum of raw output lengths across all attempts.
        """
        return sum(len(a.raw_output) for a in self.failed_attempts)


def build_reask_prompt(
    original_prompt: str,
    failed_output: str,
    error: BaseException,
    schema: dict[str, Any] | None = None,
) -> str:
    """Build a reask prompt that sends the failed output + error back to the model.

    The reask prompt appends the model's previous (invalid) attempt and the
    validation error to the original prompt, instructing the model to correct
    its output.

    Args:
        original_prompt: The original user prompt.
        failed_output: The raw output string that failed validation.
        error: The exception raised by schema validation.
        schema: Optional JSON schema dict for context.

    Returns:
        Modified prompt string for the next generation attempt.

    Example::

        prompt = build_reask_prompt("What is 2+2?", "not-json", ValueError("bad"))
        assert "PREVIOUS ATTEMPT" in prompt
    """
    schema_hint = ""
    if schema:
        import json as _json

        try:
            schema_hint = f"\n\nRequired output schema:\n{_json.dumps(schema, indent=2)}"
        except (TypeError, ValueError):
            schema_hint = ""

    return (
        f"{original_prompt}{schema_hint}\n\n"
        "---\n"
        "PREVIOUS ATTEMPT (invalid — do NOT repeat this):\n"
        f"{failed_output}\n\n"
        f"VALIDATION ERROR: {error}\n\n"
        "Please generate a corrected response that fixes the validation error above. "
        "Output ONLY the corrected JSON, nothing else."
    )


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
