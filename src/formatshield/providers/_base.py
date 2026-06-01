"""BaseProvider abstract class for shared provider logic.

Implements common functionality: retry with exponential backoff, logging,
and chars_per_token caching. Subclasses implement provider-specific logic
via abstract methods.
"""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from formatshield.exceptions import ProviderError

if TYPE_CHECKING:
    from collections.abc import Awaitable

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_RETRY_ATTEMPTS: int = 3
_DEFAULT_BACKOFF_BASE: float = 2.0
_DEFAULT_BACKOFF_MAX: float = 30.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BaseProvider ABC
# ---------------------------------------------------------------------------


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    Implements shared retry logic with exponential backoff + jitter,
    logging, and token count caching. Subclasses must implement:
      - _raw_complete(messages, max_tokens) -> str
      - _raw_count_tokens(text) -> int
      - _get_client() -> provider-specific client

    Attributes:
        model_name: Name of the model.
        _chars_per_token_cache: Cached measurement of chars per token (None until measured).

    """

    def __init__(
        self,
        model_name: str,
        *,
        max_retries: int = _DEFAULT_RETRY_ATTEMPTS,
        backoff_base: float = _DEFAULT_BACKOFF_BASE,
        backoff_max: float = _DEFAULT_BACKOFF_MAX,
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

        self.model_name = model_name
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._context_window = context_window
        self._max_output_tokens = max_output_tokens
        self._chars_per_token_cache: float | None = None

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
    async def _raw_count_tokens(self, text: str) -> int:
        """Provider-specific raw token count call.

        Args:
            text: Text to count.

        Returns:
            Token count.

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

    # --- Public API (with retry + caching) ---

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
            self._raw_complete(messages, max_tokens=max_tokens),
            operation_name="complete",
        )
        return result

    async def count_tokens(self, text: str) -> int:
        """Count tokens with retry logic.

        Args:
            text: Text to count.

        Returns:
            Token count.

        Raises:
            ProviderError: After max retries or on non-transient failure.
        """
        result: int = await self._retry_with_backoff(
            self._raw_count_tokens(text),
            operation_name="count_tokens",
        )
        return result

    # --- Caching ---

    @property
    def chars_per_token(self) -> float | None:
        """Cached measurement of characters per token.

        Returns None until set by measure_chars_per_token().
        """
        return self._chars_per_token_cache

    def set_chars_per_token(self, value: float) -> None:
        """Set the cached chars_per_token measurement.

        Args:
            value: Characters per token ratio.
        """
        if value <= 0:
            raise ValueError(f"chars_per_token must be positive, got {value}")
        self._chars_per_token_cache = value

    # --- Retry logic ---

    async def _retry_with_backoff(
        self,
        awaitable: Awaitable[T],
        *,
        operation_name: str = "operation",
    ) -> T:
        """Execute with exponential backoff retry.

        Args:
            awaitable: Awaitable to retry.
            operation_name: Name for logging.

        Returns:
            Result of the awaitable.

        Raises:
            ProviderError: After max retries.
        """
        last_error: ProviderError | None = None

        for attempt in range(self._max_retries):
            try:
                result = await awaitable
                if attempt > 0:
                    logger.info(f"{operation_name} succeeded after {attempt} retries")
                return result
            except ProviderError as e:
                last_error = e
                if not e.retryable or attempt == self._max_retries - 1:
                    # Non-retryable or final attempt: raise
                    logger.error(
                        f"{operation_name} failed (attempt {attempt + 1}/{self._max_retries}): {e}"
                    )
                    raise
                # Retryable: backoff and retry
                backoff = min(
                    self._backoff_base**attempt + random.uniform(0, 1),
                    self._backoff_max,
                )
                logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{self._max_retries}), "
                    f"retrying in {backoff:.2f}s: {e}"
                )
                await asyncio.sleep(backoff)

        # Should never reach here, but satisfy type checker
        if last_error is not None:
            raise last_error
        raise ProviderError(f"{operation_name} failed after {self._max_retries} attempts")
