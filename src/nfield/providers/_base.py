"""BaseProvider abstract class for shared provider logic.

Implements the shared retry-with-backoff and logging; subclasses implement the
provider-specific completion call and client via abstract methods.
"""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from nfield.exceptions import ProviderError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Floor for direct provider use; the engine sets this from
# ExtractionConfig.max_api_retries. 10 attempts (each honoring Retry-After) outlast a
# rolling-window TPM storm; too few surrenders a field while the minute is still capped.
_DEFAULT_RETRY_ATTEMPTS: int = 10
_DEFAULT_BACKOFF_BASE: float = 2.0
_DEFAULT_BACKOFF_MAX: float = 60.0
# A TPM 429's Retry-After reports when the FULL token window resets (~60s), but a
# token bucket refills continuously at limit/60 tokens per second, so one call's
# tokens free up in a few seconds - not a whole window. Cap the rate-limit wait
# here so throughput tracks the steady-state limit instead of sleeping a full
# window per throttled call. The attempt still counts, so a genuinely exhausted
# quota still backs off across the retry budget.
_DEFAULT_RATE_LIMIT_BACKOFF_MAX: float = 8.0
# Jitter (s) added to a server Retry-After wait so concurrent calls don't retry in
# lockstep. Added, not full jitter: it must never undercut the server's requested wait.
_RETRY_AFTER_JITTER_MAX: float = 1.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BaseProvider ABC
# ---------------------------------------------------------------------------


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    Implements shared retry logic with exponential backoff + jitter and logging.
    Subclasses must implement:
      - _raw_complete(messages, max_tokens) -> str
      - _get_client() -> provider-specific client

    Attributes:
        model_name: Name of the model.
    """

    def __init__(
        self,
        model_name: str,
        *,
        max_retries: int = _DEFAULT_RETRY_ATTEMPTS,
        backoff_base: float = _DEFAULT_BACKOFF_BASE,
        backoff_max: float = _DEFAULT_BACKOFF_MAX,
        rate_limit_backoff_max: float = _DEFAULT_RATE_LIMIT_BACKOFF_MAX,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            model_name: Name of the model (e.g., "gpt-4", "llama-3.1-8b").
            max_retries: Maximum number of retry attempts on transient failure.
                Must be > 0.
            backoff_base: Base for exponential backoff (seconds). Must be > 0.
            backoff_max: Maximum backoff duration (seconds). Must be >= backoff_base.
            rate_limit_backoff_max: Cap on the wait for a rate-limit (429)
                Retry-After (seconds). Must be > 0. Short because a TPM bucket
                refills continuously; see ``_DEFAULT_RATE_LIMIT_BACKOFF_MAX``.
            context_window: Total context window size in tokens (input + output).
                If None, uses provider-specific default or conservative 8192.
            max_output_tokens: Maximum output tokens for a single API call.
                If None, uses provider-specific default or conservative 8192.

        Raises:
            ValueError: If retry or backoff parameters are invalid.
        """
        # Validate retry and backoff parameters
        if max_retries <= 0:
            raise ValueError(f"max_retries must be > 0, got {max_retries}")
        if backoff_base <= 0:
            raise ValueError(f"backoff_base must be > 0, got {backoff_base}")
        if backoff_max < backoff_base:
            raise ValueError(
                f"backoff_max ({backoff_max}) must be >= backoff_base ({backoff_base})"
            )
        if rate_limit_backoff_max <= 0:
            raise ValueError(f"rate_limit_backoff_max must be > 0, got {rate_limit_backoff_max}")

        self.model_name = model_name
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._rate_limit_backoff_max = rate_limit_backoff_max
        self._context_window = context_window
        self._max_output_tokens = max_output_tokens

    # --- Abstract methods (subclasses must implement) ---

    @abstractmethod
    async def _raw_complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        """Provider-specific raw completion call.

        Args:
            messages: Message list.
            max_tokens: Maximum tokens.

        Returns:
            Generated text.

        Raises:
            ProviderError: On API failure.
        """
        ...

    @abstractmethod
    def _get_client(self) -> Any:
        """Get or initialize the provider-specific client.

        Returns:
            Client instance (type depends on provider).

        Raises:
            ProviderError: On client initialization failure.
        """
        ...

    # --- Abstract properties (subclasses must implement) ---

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Total context window size (input + output)."""
        ...

    @property
    @abstractmethod
    def max_output_tokens(self) -> int:
        """Maximum output tokens for a single call."""
        ...

    # --- Public API (with retry) ---

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        """Complete messages with retry logic.

        Args:
            messages: Message list.
            max_tokens: Maximum tokens.

        Returns:
            Generated text.

        Raises:
            ProviderError: After max retries or on non-transient failure.
        """
        result: str = await self._retry_with_backoff(
            lambda: self._raw_complete(messages, max_tokens=max_tokens),
            operation_name="complete",
        )
        return result

    # --- Retry logic ---

    async def _retry_with_backoff(
        self,
        factory: Callable[[], Awaitable[T]],
        *,
        operation_name: str = "operation",
    ) -> T:
        """Execute with exponential backoff retry on transient (retryable) errors.

        Takes a *factory* that produces a fresh awaitable per attempt - a coroutine
        can only be awaited once, so each retry must call the API anew. Retries only
        ``ProviderError.retryable`` failures (429, 5xx, timeouts): a server ``Retry-After``
        is honored (capped) with a small decorrelation jitter, otherwise the wait is
        exponential backoff with **full jitter**; permanent errors raise immediately
        (transient-vs-permanent classification, AWS/Google retry guidance).

        Args:
            factory: Zero-arg callable returning a fresh awaitable for each attempt.
            operation_name: Name for logging.

        Returns:
            Result of the awaitable.

        Raises:
            ProviderError: On a non-retryable error or after max retries.
        """
        last_error: ProviderError | None = None

        for attempt in range(self._max_retries):
            try:
                result = await factory()
                if attempt > 0:
                    logger.info(f"{operation_name} succeeded after {attempt} retries")
                return result
            except ProviderError as e:  # noqa: PERF203
                last_error = e
                if not e.retryable or attempt == self._max_retries - 1:
                    # Non-retryable or final attempt: raise
                    logger.error(
                        f"{operation_name} failed (attempt {attempt + 1}/{self._max_retries}): {e}"
                    )
                    raise
                # Honor a server Retry-After (capped) + small jitter - NOT full jitter,
                # since we must not retry before the server is ready. Otherwise full
                # jitter: a uniform wait in [0, exponential ceiling] spreads concurrent
                # retries and minimizes collisions (AWS Exponential Backoff and Jitter).
                if e.retry_after is not None:
                    ceiling = min(e.retry_after, self._rate_limit_backoff_max)
                    backoff = ceiling + random.uniform(0, _RETRY_AFTER_JITTER_MAX)
                else:
                    ceiling = min(self._backoff_base**attempt, self._backoff_max)
                    backoff = random.uniform(0, ceiling)
                logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{self._max_retries}), "
                    f"retrying in {backoff:.2f}s: {e}"
                )
                await asyncio.sleep(backoff)

        # Should never reach here, but satisfy type checker
        if last_error is not None:
            raise last_error
        raise ProviderError(f"{operation_name} failed after {self._max_retries} attempts")
