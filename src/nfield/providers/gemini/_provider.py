"""Native Google Gemini provider.

Talks to the Gemini API through the google-genai SDK (not an OpenAI-compatible
shim), so it uses Gemini's own request shape: system messages become the system
instruction, remaining turns become typed content, and the reply's token usage is
read back for calibration. The SDK is imported lazily, only on the first call.
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

# Defaults track the current Gemini family: 1M context and 64K output (2.5 Flash
# and Pro). Pass the real context_window / max_output_tokens for another model.
_DEFAULT_GEMINI_CONTEXT_WINDOW: int = 1_048_576
_DEFAULT_GEMINI_MAX_OUTPUT_TOKENS: int = 65_536

# HTTP statuses worth retrying: request timeout, conflict, rate limit, and the 5xx
# server family (transient-vs-permanent classification, Google API retry guidance).
_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504})
_TRANSIENT_ERROR_KEYWORDS: tuple[str, ...] = ("timed out", "timeout", "connection")


def _is_retryable(exc: Exception, status_code: int | None) -> bool | None:
    """Whether *exc* is a transient failure worth retrying.

    Returns ``True`` for a retryable HTTP status or a timeout/connection error,
    else ``None`` to defer to status-code classification - never ``False``, so a
    status-coded error is still judged by its code.

    Args:
        exc: The exception raised by the google-genai SDK.
        status_code: The HTTP status parsed from the error, if any.

    Returns:
        ``True`` if clearly transient, else ``None``.
    """
    if status_code in _RETRYABLE_STATUS:
        return True
    message = str(exc).lower()
    if any(keyword in message for keyword in _TRANSIENT_ERROR_KEYWORDS):
        return True
    return None


def _is_thinking_unsupported(exc: Exception) -> bool:
    """Whether the error is the endpoint rejecting the thinking-off parameter.

    Args:
        exc: The exception raised by the google-genai SDK.

    Returns:
        ``True`` when a 400 names the thinking budget, so the caller can retry
        once without it.
    """
    code = getattr(exc, "code", None)
    message = str(exc).lower()
    return code == 400 and "thinking" in message


# ---------------------------------------------------------------------------
# GeminiProvider class
# ---------------------------------------------------------------------------


class GeminiProvider(BaseProvider):
    """LLM provider for Google Gemini via the google-genai SDK.

    Reads the key from ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) when ``api_key``
    is not given. The synchronous SDK client is used in a worker thread so it
    stays loop-independent under the sync engine wrapper. Model specs are
    caller-supplied; conservative defaults apply otherwise.

    Attributes:
        model_name: Name of the model (e.g., "gemini-2.5-flash").
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
        """Initialize the Gemini provider.

        Args:
            model_name: Gemini model name, e.g. ``"gemini-2.5-flash"``.
            context_window: Real context window in tokens. None keeps a
                conservative default; pass the real value on large models.
            max_output_tokens: Real output ceiling in tokens. None keeps default.
            max_retries: Transient-failure retry budget. None keeps the base
                default.
            api_key: Gemini API key. None (default) reads ``GEMINI_API_KEY`` (or
                ``GOOGLE_API_KEY``) from the environment. Never logged.
            base_url: Override the API endpoint. None uses the SDK default.
            reasoning_model: When True, turn Gemini's thinking off per call so it
                does not consume the answer's output budget.

        Example:
            >>> provider = GeminiProvider("gemini-2.5-flash")  # key from env
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
        # GEMINI_API_KEY / GOOGLE_API_KEY from env and uses its default endpoint.
        # Never logged, never placed in an error message.
        self._api_key = api_key
        self._base_url = base_url
        self._reasoning_model = reasoning_model
        # Latched once the endpoint rejects the thinking-off parameter.
        self._thinking_unsupported = False

    def _get_client(self) -> Any:
        """Get or lazily create the google-genai client.

        Returns:
            ``google.genai.Client`` instance.

        Raises:
            ProviderError: If google-genai is unavailable or client init fails.
        """
        if self._client is not None:
            return self._client

        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ProviderError(
                "google-genai SDK not installed. Install it with: pip install nfield[google]"
            ) from e

        try:
            http_options = types.HttpOptions(base_url=self._base_url) if self._base_url else None
            # api_key=None lets the SDK read GEMINI_API_KEY / GOOGLE_API_KEY from env.
            self._client = genai.Client(api_key=self._api_key, http_options=http_options)
        except Exception as e:
            raise ProviderError(
                f"Failed to initialize Gemini client: {e}. "
                "Set GEMINI_API_KEY in the environment or pass api_key=..."
            ) from e

        return self._client

    # --- Abstract method implementations ---

    async def _create(self, client: Any, request: dict[str, Any]) -> Any:
        """Run the blocking generate_content call in a worker thread.

        The sync client is loop-independent (survives the sync wrapper's per-call
        ``asyncio.run``), so concurrent leaves run in parallel under the engine's
        semaphore.
        """
        return await asyncio.to_thread(client.models.generate_content, **request)

    async def _raw_complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        """Call the Gemini generate_content API.

        Args:
            messages: Message list in role/content form.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text.

        Raises:
            ProviderError: On API call failure.
        """
        from google.genai import types

        client = self._get_client()

        system_texts = [
            m["content"] for m in messages if m.get("role") == "system" and m.get("content")
        ]
        contents = [
            types.Content(
                role="model" if m.get("role") == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages
            if m.get("role") != "system"
        ]

        def _build_config(*, with_thinking_off: bool) -> Any:
            kwargs: dict[str, Any] = {"max_output_tokens": max_tokens}
            if system_texts:
                kwargs["system_instruction"] = "\n\n".join(system_texts)
            if with_thinking_off:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            return types.GenerateContentConfig(**kwargs)

        thinking_off = self._reasoning_model and not self._thinking_unsupported
        request: dict[str, Any] = {
            "model": self.model_name,
            "contents": contents,
            "config": _build_config(with_thinking_off=thinking_off),
        }
        try:
            try:
                response = await self._create(client, request)
            except Exception as e:
                if not (thinking_off and _is_thinking_unsupported(e)):
                    raise
                # The model rejects thinking_budget: stop sending it and retry once,
                # relying on the output strip instead.
                self._thinking_unsupported = True
                request["config"] = _build_config(with_thinking_off=False)
                response = await self._create(client, request)
        except Exception as e:
            status_code = getattr(e, "code", None)
            if not isinstance(status_code, int):
                status_code = None
            raise ProviderError(
                f"Gemini API call failed: {e}",
                status_code=status_code,
                retryable=_is_retryable(e, status_code),
            ) from e

        usage = getattr(response, "usage_metadata", None)
        self._record_usage(
            getattr(usage, "prompt_token_count", None) if usage else None,
            getattr(usage, "candidates_token_count", None) if usage else None,
        )
        # response.text raises if the reply was blocked or has no text part.
        try:
            text = response.text
        except Exception as e:
            raise ProviderError(f"Gemini returned no usable text: {e}", retryable=None) from e
        return strip_reasoning(text or "")

    # --- Properties ---

    @property
    def context_window(self) -> int:
        """Total context window size (input + output) in tokens.

        Returns:
            The caller-provided value, else the Gemini family default (1048576).
        """
        if self._context_window is not None:
            return self._context_window
        return _DEFAULT_GEMINI_CONTEXT_WINDOW

    @property
    def max_output_tokens(self) -> int:
        """Maximum output tokens for a single API call.

        Returns:
            The caller-provided value, else the Gemini family default (65536).
        """
        if self._max_output_tokens is not None:
            return self._max_output_tokens
        return _DEFAULT_GEMINI_MAX_OUTPUT_TOKENS
