"""Groq LLM provider implementation.

Connects to the Groq API for fast inference. Implements the LLMProvider
protocol with deferred imports (groq SDK only imported when needed).
"""

from __future__ import annotations

from typing import Any

from formatshield.exceptions import ProviderError
from formatshield.providers._base import BaseProvider

# ---------------------------------------------------------------------------
# Default model specifications
# ---------------------------------------------------------------------------

# Conservative defaults for unknown Groq models. Model specs vary by model
# and change over time. Users should provide context_window and max_output_tokens
# in the constructor for accurate specs with specific models.
_DEFAULT_GROQ_CONTEXT_WINDOW: int = 8_192
_DEFAULT_GROQ_MAX_OUTPUT_TOKENS: int = 8_192

# Groq/OpenAI-SDK exception class names for transient network failures that carry
# no HTTP status, plus message keywords as a fallback. These are retryable even
# though status_code is None.
_TRANSIENT_ERROR_NAMES: frozenset[str] = frozenset(
    {"APITimeoutError", "APIConnectionError", "InternalServerError"}
)
_TRANSIENT_ERROR_KEYWORDS: tuple[str, ...] = ("timed out", "timeout", "connection")


def _is_transient_error(exc: Exception) -> bool | None:
    """Whether *exc* is a transient network failure that should be retried.

    Returns ``True`` for timeout/connection errors (which carry no HTTP status),
    or ``None`` to defer to status-code classification when undeterminable — never
    ``False``, so a status-coded error is still judged by its code.

    Args:
        exc: The exception raised by the Groq SDK.

    Returns:
        ``True`` if clearly transient, else ``None``.
    """
    if type(exc).__name__ in _TRANSIENT_ERROR_NAMES:
        return True
    message = str(exc).lower()
    if any(keyword in message for keyword in _TRANSIENT_ERROR_KEYWORDS):
        return True
    return None


def _retry_after_seconds(exc: Exception) -> float | None:
    """Extract the ``Retry-After`` delay (seconds) from a provider error, if present.

    Args:
        exc: The exception raised by the Groq SDK.

    Returns:
        The delay in seconds, or ``None`` when absent or not a plain number (an
        HTTP-date form is ignored — the backoff loop falls back to its own timing).
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# GroqProvider class
# ---------------------------------------------------------------------------


class GroqProvider(BaseProvider):
    """Groq LLM provider implementation.

    Connects to Groq's API for low-latency inference. The groq SDK is
    imported lazily only when a call is made, not at import time.

    Allows users to override model specs (context_window, max_output_tokens)
    for flexibility with new or updated models.

    Attributes:
        model_name: Name of the Groq model (e.g., "llama-3.1-8b").
    """

    def __init__(
        self,
        model_name: str,
        *,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        """Initialize Groq provider.

        Args:
            model_name: Groq model name (e.g., "llama-3.1-8b").
            context_window: Total context window size in tokens. If None, uses
                default 8192. Provide this if you know the actual context window
                for your model.
            max_output_tokens: Maximum output tokens. If None, uses default 8192.
                Provide this if you know the actual limit for your model.

        Example:
            >>> # Use defaults for unknown model
            >>> provider = GroqProvider("llama-3.2-new")
            >>>
            >>> # Override with known specs
            >>> provider = GroqProvider(
            ...     "llama-3.1-70b",
            ...     context_window=131_072,
            ...     max_output_tokens=8_192,
            ... )
        """
        super().__init__(
            model_name,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
        )
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or initialize the Groq client.

        Performs lazy import of the groq SDK. Raises ProviderError if
        the groq package is not installed or if API credentials are missing.

        Returns:
            Groq client instance.

        Raises:
            ProviderError: If groq SDK unavailable or credentials missing.
        """
        if self._client is not None:
            return self._client

        try:
            # Deferred import: only import groq when we actually need it
            import groq
        except ImportError as e:
            raise ProviderError(
                "groq SDK not installed. Install it with: pip install formatshield[groq]"
            ) from e

        try:
            self._client = groq.Groq()
        except Exception as e:
            raise ProviderError(
                f"Failed to initialize Groq client: {e}. "
                "Make sure GROQ_API_KEY is set in environment variables."
            ) from e

        return self._client

    # --- Abstract method implementations ---

    async def _raw_complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        """Call Groq API for text completion.

        Args:
            messages: Message list in OpenAI format.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text.

        Raises:
            ProviderError: On API call failure.
        """
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            status_code = getattr(e, "status_code", None)
            raise ProviderError(
                f"Groq API call failed: {e}",
                status_code=status_code,
                retryable=_is_transient_error(e),
                retry_after=_retry_after_seconds(e),
            ) from e

    async def _raw_count_tokens(self, text: str) -> int:
        """Count tokens using Groq's tokenization.

        Note: Groq SDK may not have a dedicated token-counting API.
        This is a stub that will call the model's tokenizer if available,
        or fall back to estimation.

        Args:
            text: Text to tokenize.

        Returns:
            Estimated token count.

        Raises:
            ProviderError: On API failure.
        """
        # Groq SDK doesn't expose a tokenization API in MVP.
        # Use a simple heuristic: assume ~3.5 chars per token for English.
        # In production, this could call a dedicated tokenizer endpoint.
        estimated_tokens = max(1, len(text) // 4)
        return estimated_tokens

    # --- Properties ---

    @property
    def context_window(self) -> int:
        """Total context window size (input + output) in tokens.

        Returns user-provided value if available, otherwise conservative default.
        Users should override this for accuracy with specific models.

        Returns:
            Context window in tokens (default 8192 if not specified).
        """
        if self._context_window is not None:
            return self._context_window
        return _DEFAULT_GROQ_CONTEXT_WINDOW

    @property
    def max_output_tokens(self) -> int:
        """Maximum output tokens for a single API call.

        Returns user-provided value if available, otherwise conservative default.
        Users should override this for accuracy with specific models.

        Returns:
            Max output tokens (default 8192 if not specified).
        """
        if self._max_output_tokens is not None:
            return self._max_output_tokens
        return _DEFAULT_GROQ_MAX_OUTPUT_TOKENS
