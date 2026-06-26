"""OpenAI-compatible LLM provider implementation.

Connects to any endpoint that speaks the OpenAI ``/v1/chat/completions`` API.
The default target is OpenAI itself; a ``base_url`` retargets the same class at
any compatible service — hosted (Together, Fireworks, OpenRouter, DeepSeek, xAI,
Mistral, Azure) or local (Ollama, vLLM, LM Studio). The openai SDK is imported
lazily, only when a call is made.
"""

from __future__ import annotations

import asyncio
from typing import Any

from nfield.exceptions import ProviderError
from nfield.providers._base import BaseProvider

# ---------------------------------------------------------------------------
# Default model specifications
# ---------------------------------------------------------------------------

# Conservative defaults for an unknown model behind an OpenAI-compatible endpoint.
# Context windows differ across OpenAI, hosted gateways, and local servers, so
# callers should pass context_window and max_output_tokens for accurate capacity
# planning with a specific model.
_DEFAULT_OPENAI_CONTEXT_WINDOW: int = 8_192
_DEFAULT_OPENAI_MAX_OUTPUT_TOKENS: int = 8_192

# openai SDK exception class names for transient network failures that carry no
# HTTP status, plus message keywords as a fallback. Retryable even though
# status_code is None. (The groq SDK is a fork of this SDK, so the names match.)
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
        exc: The exception raised by the openai SDK.

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
        exc: The exception raised by the openai SDK.

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
# OpenAIProvider class
# ---------------------------------------------------------------------------


class OpenAIProvider(BaseProvider):
    """LLM provider for the OpenAI ``/v1/chat/completions`` API and its clones.

    Talks to OpenAI by default; set ``base_url`` to reach any OpenAI-compatible
    endpoint with the same class. The openai SDK is imported lazily only when a
    call is made, not at import time. The synchronous SDK client is used (run in a
    worker thread) so it stays loop-independent under the sync engine wrapper.

    Model specs (context_window, max_output_tokens) are caller-supplied so
    capacity planning matches the real model; conservative defaults apply
    otherwise.

    Attributes:
        model_name: Name of the model (e.g., "gpt-4o-mini").
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
        """Initialize the OpenAI-compatible provider.

        Args:
            model_name: Model name as the endpoint expects it (e.g., "gpt-4o-mini",
                "meta-llama/Llama-3.1-8B-Instruct", "llama3.2").
            context_window: Total context window size in tokens. If None, uses
                default 8192. Provide this for accurate capacity planning.
            max_output_tokens: Maximum output tokens. If None, uses default 8192.
                Provide this if you know the actual limit for your model.
            max_retries: Transient-failure retry budget per call. If None, the
                base provider default applies.
            api_key: API key. If None (default), the SDK reads ``OPENAI_API_KEY``
                from the environment — the recommended path. Pass it explicitly
                only for secret-vault / multi-tenant setups, or to supply the key
                of a non-OpenAI compatible service. It is stored solely to build
                the client and is never logged or echoed in errors. Local servers
                (Ollama, vLLM) accept any non-empty placeholder.
            base_url: Override the API base URL to target an OpenAI-compatible
                endpoint. If None, the SDK default (OpenAI). Examples:
                ``"https://api.together.xyz/v1"``, ``"https://api.deepseek.com"``,
                ``"http://localhost:11434/v1"`` (Ollama),
                ``"http://localhost:8000/v1"`` (vLLM).

        Example:
            >>> # OpenAI, key from OPENAI_API_KEY env
            >>> provider = OpenAIProvider("gpt-4o-mini")
            >>>
            >>> # A local model served by vLLM
            >>> provider = OpenAIProvider(
            ...     "meta-llama/Llama-3.1-8B-Instruct",
            ...     context_window=131_072,
            ...     api_key="not-needed",
            ...     base_url="http://localhost:8000/v1",
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
        # openai SDK uses its own default (api_key from OPENAI_API_KEY env; standard
        # base URL). Never logged, never placed in an error message.
        self._api_key = api_key
        self._base_url = base_url

    def _get_client(self) -> Any:
        """Get or initialize the OpenAI client.

        Performs a lazy import of the openai SDK. Raises ProviderError if the
        openai package is not installed or the client cannot be constructed.

        Returns:
            ``openai.OpenAI`` client instance.

        Raises:
            ProviderError: If openai SDK unavailable or client init fails.
        """
        if self._client is not None:
            return self._client

        try:
            # Deferred import: only import openai when we actually need it.
            import openai
        except ImportError as e:
            raise ProviderError(
                "openai SDK not installed. Install it with: pip install nfield[openai]"
            ) from e

        try:
            # Pass api_key/base_url through; None lets the SDK fall back to its
            # defaults (OPENAI_API_KEY env, standard base URL).
            self._client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
        except Exception as e:
            raise ProviderError(
                f"Failed to initialize OpenAI client: {e}. "
                "Set OPENAI_API_KEY in the environment or pass api_key=..."
            ) from e

        return self._client

    # --- Abstract method implementations ---

    async def _raw_complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        """Call the OpenAI-compatible chat completions API.

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
            # The sync openai client is loop-independent, so it survives the sync
            # wrapper's per-call asyncio.run (an AsyncOpenAI client would bind to one
            # loop and warn once it closed). Run the blocking call in a worker thread
            # so concurrent leaf calls from asyncio.gather don't serialize; httpx.Client
            # is thread-safe and the engine's semaphore bounds the thread count.
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
                f"OpenAI API call failed: {e}",
                status_code=status_code,
                retryable=_is_transient_error(e),
                retry_after=_retry_after_seconds(e),
            ) from e

    # --- Properties ---

    @property
    def context_window(self) -> int:
        """Total context window size (input + output) in tokens.

        Returns the user-provided value if available, otherwise a conservative
        default. Override this for accuracy with a specific model.

        Returns:
            Context window in tokens (default 8192 if not specified).
        """
        if self._context_window is not None:
            return self._context_window
        return _DEFAULT_OPENAI_CONTEXT_WINDOW

    @property
    def max_output_tokens(self) -> int:
        """Maximum output tokens for a single API call.

        Returns the user-provided value if available, otherwise a conservative
        default. Override this for accuracy with a specific model.

        Returns:
            Max output tokens (default 8192 if not specified).
        """
        if self._max_output_tokens is not None:
            return self._max_output_tokens
        return _DEFAULT_OPENAI_MAX_OUTPUT_TOKENS
