"""Native Anthropic (Claude) provider.

Talks to the Claude Messages API through the anthropic SDK. System messages are
sent as the top-level ``system`` parameter (Claude keeps system text out of the
message list), and the reply's input-token usage is read back for calibration.
The SDK is imported lazily, only on the first call.
"""

from __future__ import annotations

import asyncio
from typing import Any

from nfield.exceptions import ProviderError
from nfield.providers._base import BaseProvider
from nfield.providers._reasoning import strip_reasoning

# ---------------------------------------------------------------------------
# Default model specifications
# ---------------------------------------------------------------------------

# Defaults track the current Claude family floor: 200K context and 64K output
# (Haiku 4.5). Pass the real context_window / max_output_tokens for a larger model.
_DEFAULT_ANTHROPIC_CONTEXT_WINDOW: int = 200_000
_DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS: int = 64_000

# anthropic SDK exception class names for transient network failures that carry no
# HTTP status, plus message keywords as a fallback. Retryable even though
# status_code is None.
_TRANSIENT_ERROR_NAMES: frozenset[str] = frozenset(
    {"APITimeoutError", "APIConnectionError", "InternalServerError"}
)
_TRANSIENT_ERROR_KEYWORDS: tuple[str, ...] = ("timed out", "timeout", "connection")

# Per-request timeout, sized to the booked output so it does not expire mid-decode:
# floor for prompt+connect, then max_tokens at a worst-case decode rate.
_REQUEST_TIMEOUT_FLOOR_S: float = 120.0
_DECODE_FLOOR_TOKENS_PER_S: float = 50.0


def _is_transient_error(exc: Exception) -> bool | None:
    """Whether *exc* is a transient network failure that should be retried.

    Returns ``True`` for timeout/connection errors (which carry no HTTP status),
    or ``None`` to defer to status-code classification - never ``False``.

    Args:
        exc: The exception raised by the anthropic SDK.

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
        exc: The exception raised by the anthropic SDK.

    Returns:
        The delay in seconds, or ``None`` when absent or not a plain number.
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
# AnthropicProvider class
# ---------------------------------------------------------------------------


class AnthropicProvider(BaseProvider):
    """LLM provider for Anthropic Claude via the anthropic SDK.

    Reads the key from ``ANTHROPIC_API_KEY`` when ``api_key`` is not given. The
    synchronous SDK client is used in a worker thread so it stays loop-independent
    under the sync engine wrapper. Model specs are caller-supplied; conservative
    defaults apply otherwise.

    Attributes:
        model_name: Name of the model (e.g., "claude-sonnet-4").
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
        """Initialize the Anthropic provider.

        Args:
            model_name: Claude model name, e.g. ``"claude-sonnet-4"``.
            context_window: Real context window in tokens. None keeps a
                conservative default; pass the real value on large models.
            max_output_tokens: Real output ceiling in tokens. None keeps default.
            max_retries: Transient-failure retry budget. None keeps the base
                default.
            api_key: Anthropic API key. None (default) reads ``ANTHROPIC_API_KEY``
                from the environment. Never logged.
            base_url: Override the API endpoint. None uses the SDK default.
            reasoning_model: Accepted for interface parity. Claude keeps extended
                thinking off unless explicitly enabled, so this is a no-op.

        Example:
            >>> provider = AnthropicProvider("claude-sonnet-4")  # key from env
        """
        super().__init__(
            model_name,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            # Override the base default only when the caller set it.
            **({} if max_retries is None else {"max_retries": max_retries}),
        )
        self._client: Any = None
        # Stored only to construct the SDK client. None means the SDK reads
        # ANTHROPIC_API_KEY from env. Never logged, never placed in an error.
        self._api_key = api_key
        self._base_url = base_url
        self._reasoning_model = reasoning_model

    def _get_client(self) -> Any:
        """Get or lazily create the anthropic client.

        Returns:
            ``anthropic.Anthropic`` client instance.

        Raises:
            ProviderError: If the anthropic SDK is unavailable or init fails.
        """
        if self._client is not None:
            return self._client

        try:
            import anthropic
        except ImportError as e:
            raise ProviderError(
                "anthropic SDK not installed. Install it with: pip install nfield[anthropic]"
            ) from e

        try:
            kwargs: dict[str, Any] = {"api_key": self._api_key, "max_retries": 0}
            if self._base_url is not None:
                kwargs["base_url"] = self._base_url
            self._client = anthropic.Anthropic(**kwargs)
        except Exception as e:
            raise ProviderError(
                f"Failed to initialize Anthropic client: {e}. "
                "Set ANTHROPIC_API_KEY in the environment or pass api_key=..."
            ) from e

        return self._client

    # --- Abstract method implementations ---

    async def _create(self, client: Any, request: dict[str, Any]) -> Any:
        """Run the blocking messages.create call in a worker thread."""
        return await asyncio.to_thread(client.messages.create, **request)

    async def _raw_complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        """Call the Claude Messages API.

        Args:
            messages: Message list in role/content form.
            max_tokens: Maximum tokens to generate (required by the API).

        Returns:
            Generated text.

        Raises:
            ProviderError: On API call failure.
        """
        client = self._get_client()

        system_texts = [
            m["content"] for m in messages if m.get("role") == "system" and m.get("content")
        ]
        turns = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") != "system"
        ]
        request: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": turns,
            "timeout": max(_REQUEST_TIMEOUT_FLOOR_S, max_tokens / _DECODE_FLOOR_TOKENS_PER_S),
        }
        # Claude takes system text as a top-level parameter, not a message.
        if system_texts:
            request["system"] = "\n\n".join(system_texts)

        try:
            response = await self._create(client, request)
        except Exception as e:
            status_code = getattr(e, "status_code", None)
            raise ProviderError(
                f"Anthropic API call failed: {e}",
                status_code=status_code,
                retryable=_is_transient_error(e),
                retry_after=_retry_after_seconds(e),
            ) from e

        usage = getattr(response, "usage", None)
        self.last_prompt_tokens = getattr(usage, "input_tokens", None) if usage else None
        # The reply is a list of content blocks; concatenate their text parts.
        text = "".join(
            getattr(block, "text", "") for block in getattr(response, "content", None) or []
        )
        return strip_reasoning(text)

    # --- Properties ---

    @property
    def context_window(self) -> int:
        """Total context window size (input + output) in tokens.

        Returns:
            The caller-provided value, else the Claude family default (200000).
        """
        if self._context_window is not None:
            return self._context_window
        return _DEFAULT_ANTHROPIC_CONTEXT_WINDOW

    @property
    def max_output_tokens(self) -> int:
        """Maximum output tokens for a single API call.

        Returns:
            The caller-provided value, else the Claude family default (64000).
        """
        if self._max_output_tokens is not None:
            return self._max_output_tokens
        return _DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS
