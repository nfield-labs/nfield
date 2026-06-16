"""Groq LLM provider implementation.

Connects to the Groq API for fast inference. Implements the LLMProvider
protocol with deferred imports (groq SDK only imported when needed).
"""

from __future__ import annotations

import asyncio
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
        max_retries: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize Groq provider.

        Args:
            model_name: Groq model name (e.g., "llama-3.1-8b").
            context_window: Total context window size in tokens. If None, uses
                default 8192. Provide this if you know the actual context window
                for your model.
            max_output_tokens: Maximum output tokens. If None, uses default 8192.
                Provide this if you know the actual limit for your model.
            max_retries: Transient-failure retry budget per call. If None, the
                base provider default applies.
            api_key: Groq API key. If None (default), the SDK reads ``GROQ_API_KEY``
                from the environment — the recommended path. Pass it explicitly only
                for secret-vault / multi-tenant setups. It is stored solely to build
                the client and is never logged or echoed in errors.
            base_url: Override the Groq API base URL (proxy, gateway, or
                Groq-compatible self-hosted endpoint). If None, the SDK default.

        Example:
            >>> # Use defaults for unknown model (key from GROQ_API_KEY env)
            >>> provider = GroqProvider("llama-3.2-new")
            >>>
            >>> # Override specs, key, and endpoint explicitly
            >>> provider = GroqProvider(
            ...     "llama-3.1-70b",
            ...     context_window=131_072,
            ...     max_output_tokens=8_192,
            ...     api_key="gsk_...",
            ...     base_url="https://my-proxy.example/v1",
            ... )
        """
        super().__init__(
            model_name,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            # Override the base default only when the caller set it.
            **({} if max_retries is None else {"max_retries": max_retries}),
        )
        self._client: Any = None
        # Stored only to construct the SDK client below. None for either means the
        # groq SDK uses its own default (api_key from GROQ_API_KEY env; standard
        # base URL). Never logged, never placed in an error message.
        self._api_key = api_key
        self._base_url = base_url

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
            # Pass api_key/base_url through; None lets the SDK fall back to its
            # defaults (GROQ_API_KEY env, standard base URL).
            self._client = groq.Groq(api_key=self._api_key, base_url=self._base_url)
        except Exception as e:
            raise ProviderError(
                f"Failed to initialize Groq client: {e}. "
                "Set GROQ_API_KEY in the environment or pass api_key=..."
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
            # The groq SDK client is synchronous, so run the blocking call in a
            # worker thread. Without this, awaiting it would still block the event
            # loop and serialize the concurrent leaf calls Stage 4/5 fire via
            # asyncio.gather — defeating max_concurrent_calls. httpx.Client (under
            # the SDK) is thread-safe, and the semaphore bounds the thread count.
            response = await asyncio.to_thread(
                client.chat.completions.create,
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
        """Estimate token count locally — the Groq SDK exposes no token API.

        IMPORTANT: this makes NO network call. Groq has no token-counting
        endpoint, so this returns a ``len // 4`` character heuristic. As a result
        Stage 0 calibration via this provider yields ~4.0 chars/token (a constant
        estimate), not a tokenizer-measured value. A future upgrade would count
        with the model's real tokenizer offline (e.g. tiktoken / transformers).

        Args:
            text: Text to estimate a token count for.

        Returns:
            Estimated token count (``len(text) // 4``, minimum 1).
        """
        return max(1, len(text) // 4)

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
