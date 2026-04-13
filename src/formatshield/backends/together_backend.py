"""Together AI inference backend for FormatShield.

Together AI exposes an OpenAI-compatible API, so this backend is a thin
wrapper around the OpenAI client pointed at Together AI's base URL.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from formatshield._retry import API_RETRY, with_retry
from formatshield.scorer.features import StreamEvent

_TOGETHER_BASE_URL = "https://api.together.xyz/v1"


class TogetherBackend:
    """
    FormatShield backend that targets the `Together AI <https://www.together.ai>`_
    hosted inference API.

    Together AI exposes an OpenAI-compatible endpoint, so this backend reuses
    the ``openai`` SDK client with a custom ``base_url``.  It does **not**
    expose server-side KV-cache prefix reuse, so :attr:`supports_kv_cache_reuse`
    is ``False``.

    Parameters
    ----------
    api_key:
        Together AI API key.  If ``None``, the value of the
        ``TOGETHER_API_KEY`` environment variable is used.  A
        :exc:`ValueError` is raised when neither source provides a key.
    model:
        Model identifier.  Accepts both plain model names (e.g.
        ``"meta-llama/Llama-3-70b-chat-hf"``) and the ``"together/model-name"``
        prefixed format used by FormatShield's router.
    """

    #: Backend identifier consumed by the FormatShield router.
    name: str = "together"

    #: Together AI does not expose server-side KV-cache prefix reuse.
    supports_kv_cache_reuse: bool = False

    #: 16% baseline accuracy loss measured on FormatShield's benchmark suite.
    accuracy_loss_baseline: float | None = 0.16

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "meta-llama/Llama-3-70b-chat-hf",
    ) -> None:
        resolved_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Together AI API key supplied.  Pass api_key= or set the "
                "TOGETHER_API_KEY environment variable."
            )
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for TogetherBackend. "
                "Install it with: pip install 'formatshield[together]' or pip install openai"
            ) from exc
        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=_TOGETHER_BASE_URL,
        )
        # Strip optional "together/" prefix so the raw model name reaches the API.
        self.model = model.removeprefix("together/")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
        constraints: str | None,
    ) -> list[dict[str, Any]]:
        """Assemble the OpenAI-compatible messages list."""
        messages: list[dict[str, Any]] = []

        if schema and constraints != "json":
            schema_text = json.dumps(schema, indent=2)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "You must respond with valid JSON that conforms to the "
                        f"following JSON schema:\n\n{schema_text}\n\n"
                        "Do not include any text outside the JSON object."
                    ),
                }
            )

        messages.append({"role": "user", "content": prompt})
        return messages

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
        kv_cache_prefix: str | None = None,
    ) -> str:
        """Generate a response and return the full text.

        When *schema* is provided **and** *constraints* is ``"json"``,
        Together AI's JSON response mode is activated. When only *schema* is
        provided, it is embedded in a system prompt.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict.
            constraints: Pass ``"json"`` to activate JSON response mode.
            kv_cache_prefix: Ignored; Together AI does not support prefix caching.

        Returns:
            The model's response text.

        Raises:
            RuntimeError: Wraps any Together AI API error with a human-readable message.
        """
        import openai  # type: ignore[import-not-found]

        del kv_cache_prefix  # Together AI does not support prefix caching
        messages = self._build_messages(prompt, schema, constraints)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }

        if constraints == "json":
            kwargs["response_format"] = {"type": "json_object"}

        async def _call() -> str:
            try:
                response = await self._client.chat.completions.create(**kwargs)
            except (openai.RateLimitError, openai.InternalServerError):
                raise  # propagate retryable errors un-wrapped
            except openai.APIError as exc:
                raise RuntimeError(f"Together AI API error: {exc}") from exc
            content = response.choices[0].message.content
            return content if content is not None else ""

        return await with_retry(
            _call,
            API_RETRY,
            retryable=(openai.RateLimitError, openai.InternalServerError),
            operation_name=f"together.generate({self.model})",
        )

    async def stream(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the model's response as StreamEvent objects.

        Yields one ``"output"`` event per incremental token chunk, then a
        single ``"complete"`` event containing the fully assembled text.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict.
            constraints: Pass ``"json"`` to activate JSON response mode.

        Yields:
            Incremental output tokens followed by a final complete event.

        Raises:
            RuntimeError: Wraps any Together AI API error with a human-readable message.
        """
        return self._stream_impl(prompt, schema, constraints)

    async def _stream_impl(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
        constraints: str | None,
    ) -> AsyncIterator[StreamEvent]:
        import openai  # type: ignore[import-not-found]

        messages = self._build_messages(prompt, schema, constraints)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "stream": True,
        }

        if constraints == "json":
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.monotonic()
        accumulated = ""

        try:
            async with await self._client.chat.completions.create(**kwargs) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        accumulated += delta
                        yield StreamEvent(
                            type="output",
                            token=delta,
                            backend=self.name,
                            latency_ms=(time.monotonic() - t0) * 1000,
                        )
        except openai.APIError as exc:
            raise RuntimeError(f"Together AI API streaming error: {exc}") from exc

        yield StreamEvent(
            type="complete",
            content=accumulated,
            backend=self.name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
