"""Groq LLM provider implementation.

Connects to the Groq API for fast inference. Implements the LLMProvider
protocol with deferred imports (groq SDK only imported when needed).
"""

from __future__ import annotations

import asyncio
from typing import Any

from nfield.exceptions import ProviderError
from nfield.providers._base import BaseProvider
from nfield.providers._reasoning import (
    is_unsupported_reasoning_param_error,
    reasoning_suppression_kwargs,
    strip_reasoning,
)

# ---------------------------------------------------------------------------
# Default model specifications
# ---------------------------------------------------------------------------

# Conservative fallback for an unknown Groq model; models vary (up to ~131K), so
# pass the real context_window / max_output_tokens for capacity planning.
_DEFAULT_GROQ_CONTEXT_WINDOW: int = 8_192
_DEFAULT_GROQ_MAX_OUTPUT_TOKENS: int = 8_192

# Per-request timeout, sized to the booked output so it does not expire mid-decode:
# floor for prompt+connect, then max_tokens at a worst-case decode rate.
_REQUEST_TIMEOUT_FLOOR_S: float = 120.0
_DECODE_FLOOR_TOKENS_PER_S: float = 50.0

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
    or ``None`` to defer to status-code classification when undeterminable - never
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
        HTTP-date form is ignored - the backoff loop falls back to its own timing).
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

    Connects to Groq's API for low-latency inference. The groq SDK is imported
    lazily only when a call is made, not at import time. Model specs are
    caller-supplied; conservative defaults apply otherwise.

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
        reasoning_model: bool = False,
    ) -> None:
        """Initialize Groq provider.

        Args:
            model_name: Groq model name (e.g., "llama-3.1-8b").
            context_window: Total context window size in tokens. If None, uses a
                conservative 8192. Pass the model's real window (e.g. 131072 for
                llama-3.3-70b) so capacity planning fills it - the small default is
                safe but packs many more, smaller calls than necessary.
            max_output_tokens: Maximum output tokens. If None, uses default 8192.
                Provide this if you know the actual limit for your model.
            max_retries: Transient-failure retry budget per call. If None, the
                base provider default applies.
            api_key: Groq API key. If None (default), the SDK reads ``GROQ_API_KEY``
                from the environment - the recommended path. Pass it explicitly only
                for secret-vault / multi-tenant setups. It is stored solely to build
                the client and is never logged or echoed in errors.
            base_url: Override the Groq API base URL (proxy, gateway, or
                Groq-compatible self-hosted endpoint). If None, the SDK default.
            reasoning_model: When True, disable the model's thinking on each call
                so it does not consume the answer's output budget. Default False.

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
        self._reasoning_model = reasoning_model
        # Latched once the endpoint rejects the thinking-off parameter.
        self._suppression_unsupported = False

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
                "groq SDK not installed. Install it with: pip install nfield[groq]"
            ) from e

        try:
            # None api_key/base_url lets the SDK use GROQ_API_KEY env + default URL.
            # max_retries=0: BaseProvider.complete is the sole retry owner.
            self._client = groq.Groq(api_key=self._api_key, base_url=self._base_url, max_retries=0)
        except Exception as e:
            raise ProviderError(
                f"Failed to initialize Groq client: {e}. "
                "Set GROQ_API_KEY in the environment or pass api_key=..."
            ) from e

        return self._client

    # --- Abstract method implementations ---

    async def _create(
        self, client: Any, base_kwargs: dict[str, Any], extra: dict[str, Any]
    ) -> Any:
        """Run the blocking chat-completions call in a worker thread.

        The groq SDK client is synchronous, so it runs off the event loop while the
        engine's semaphore bounds the concurrent leaf calls Stage 4/5 fire.
        """
        return await asyncio.to_thread(client.chat.completions.create, **base_kwargs, **extra)

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
        base_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "timeout": max(_REQUEST_TIMEOUT_FLOOR_S, max_tokens / _DECODE_FLOOR_TOKENS_PER_S),
        }
        # Turn thinking off for a declared reasoning model so it does not consume
        # the answer's output budget.
        suppression = (
            reasoning_suppression_kwargs(self._base_url)
            if self._reasoning_model and not self._suppression_unsupported
            else {}
        )
        try:
            try:
                response = await self._create(client, base_kwargs, suppression)
            except Exception as e:
                if not (suppression and is_unsupported_reasoning_param_error(e)):
                    raise
                # The endpoint rejects the thinking-off parameter: stop sending it and
                # retry once without, relying on the output strip instead.
                self._suppression_unsupported = True
                response = await self._create(client, base_kwargs, {})
        except Exception as e:
            status_code = getattr(e, "status_code", None)
            raise ProviderError(
                f"Groq API call failed: {e}",
                status_code=status_code,
                retryable=_is_transient_error(e),
                retry_after=_retry_after_seconds(e),
            ) from e

        usage = getattr(response, "usage", None)
        self.last_prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        return strip_reasoning(response.choices[0].message.content or "")

    # --- Properties ---

    @property
    def context_window(self) -> int:
        """Total context window size (input + output) in tokens.

        Returns:
            The caller-provided value, else a conservative default (8192).
        """
        if self._context_window is not None:
            return self._context_window
        return _DEFAULT_GROQ_CONTEXT_WINDOW

    @property
    def max_output_tokens(self) -> int:
        """Maximum output tokens for a single API call.

        Returns:
            The caller-provided value, else a conservative default (8192).
        """
        if self._max_output_tokens is not None:
            return self._max_output_tokens
        return _DEFAULT_GROQ_MAX_OUTPUT_TOKENS
