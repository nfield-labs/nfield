"""Fireworks AI inference backend for FormatShield."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from formatshield._retry import API_RETRY, with_retry
from formatshield.scorer.features import StreamEvent

#: Default Fireworks AI model identifier.
DEFAULT_FIREWORKS_MODEL = "accounts/fireworks/models/llama-v3p1-70b-instruct"

#: Fireworks AI OpenAI-compatible API base URL.
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


class FireworksBackend:
    """Fireworks AI inference backend using the OpenAI-compatible REST API.

    Fireworks AI exposes an OpenAI-compatible chat-completions endpoint at
    ``https://api.fireworks.ai/inference/v1``.  This backend uses the
    ``openai`` SDK with a custom ``base_url`` so no additional
    Fireworks-specific package is required.

    Args:
        api_key: Fireworks API key.  If ``None``, the value of the
            ``FIREWORKS_API_KEY`` environment variable is used.  A
            :exc:`ValueError` is raised when neither source provides a key.
        model: Model identifier.  Accepts both plain Fireworks model names
            (e.g. ``"accounts/fireworks/models/llama-v3p1-70b-instruct"``)
            and the ``"fireworks/<model>"`` prefixed format used by the
            FormatShield router.
    """

    #: Backend identifier consumed by the FormatShield router.
    name: str = "fireworks"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_FIREWORKS_MODEL,
    ) -> None:
        resolved_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Fireworks API key supplied.  Pass api_key= or set the "
                "FIREWORKS_API_KEY environment variable."
            )
        self._api_key = resolved_key
        self.model = model.removeprefix("fireworks/")
        self._client: Any = None

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supports_kv_cache_reuse(self) -> bool:
        """Fireworks AI does not expose server-side KV-cache prefix reuse."""
        return False

    @property
    def accuracy_loss_baseline(self) -> float | None:
        """13 % baseline accuracy loss for structured-output generation on Fireworks AI."""
        return 0.13

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return (and lazily create) the OpenAI async client pointed at Fireworks AI.

        Raises:
            ImportError: If ``openai`` is not installed.
        """
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai is required for FireworksBackend. Install with: pip install openai"
                ) from exc
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=FIREWORKS_BASE_URL,
            )
        return self._client

    def _build_messages(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
        constraints: str | None,
    ) -> list[dict[str, Any]]:
        """Assemble the OpenAI-compatible messages list.

        When a JSON schema is provided or constraints is ``"json"``, a system
        message is prepended that instructs the model to return valid JSON.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict.
            constraints: Pass ``"json"`` to activate JSON mode.

        Returns:
            A list of message dicts suitable for the chat completions API.
        """
        messages: list[dict[str, Any]] = []

        if constraints == "json" or schema:
            if schema:
                schema_text = json.dumps(schema, indent=2)
                system_content = (
                    "Respond with valid JSON that conforms to the following "
                    f"JSON schema:\n\n{schema_text}\n\n"
                    "Return only the JSON object with no surrounding text."
                )
            else:
                system_content = (
                    "Respond with valid JSON only. "
                    "Return only the JSON object with no surrounding text."
                )
            messages.append({"role": "system", "content": system_content})

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
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: list[str] | str | None = None,
    ) -> str:
        """Generate a response via the Fireworks AI chat-completions endpoint.

        When *schema* is provided or *constraints* is ``"json"``,
        ``response_format={"type": "json_object"}`` is sent to activate
        Fireworks JSON mode.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict.
            constraints: Pass ``"json"`` to activate JSON mode.
            kv_cache_prefix: Ignored; Fireworks AI does not support prefix
                caching.
            temperature: Sampling temperature.  Defaults to ``0`` for
                deterministic output.
            top_p: Nucleus sampling probability.  ``None`` defers to the API
                default.
            top_k: Ignored; not exposed by the Fireworks AI API.
            max_tokens: Maximum tokens to generate.  ``None`` defers to the
                API default.
            seed: Random seed for reproducible sampling.  ``None`` defers to
                the API default.
            frequency_penalty: Frequency penalty.  ``None`` defers to the API
                default.
            presence_penalty: Presence penalty.  ``None`` defers to the API
                default.
            stop: Stop sequence(s).  ``None`` defers to the API default.

        Returns:
            The model's response as a plain string.

        Raises:
            ImportError: If ``openai`` is not installed.
            RuntimeError: If the API call fails after retries.
        """
        import openai

        messages = self._build_messages(prompt, schema, constraints)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if seed is not None:
            kwargs["seed"] = seed
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
        if stop is not None:
            kwargs["stop"] = stop
        if constraints == "json" or schema:
            kwargs["response_format"] = {"type": "json_object"}

        client = self._get_client()

        async def _call() -> str:
            try:
                response = await client.chat.completions.create(**kwargs)
            except (openai.RateLimitError, openai.InternalServerError):
                raise
            except openai.APIError as exc:
                raise RuntimeError(f"Fireworks API error: {exc}") from exc
            content = response.choices[0].message.content
            return content if content is not None else ""

        return await with_retry(
            _call,
            API_RETRY,
            retryable=(openai.RateLimitError, openai.InternalServerError),
            operation_name=f"fireworks.generate({self.model})",
        )

    async def stream(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: list[str] | str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the model's response as :class:`StreamEvent` objects.

        Yields one ``"output"`` event per incremental token chunk, then a
        single ``"complete"`` event containing the fully assembled text.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict.
            constraints: Pass ``"json"`` to activate JSON mode.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Ignored.
            max_tokens: Maximum tokens to generate.
            seed: Random seed.
            frequency_penalty: Frequency penalty.
            presence_penalty: Presence penalty.
            stop: Stop sequence(s).

        Yields:
            Incremental :class:`StreamEvent` objects followed by a final
            ``"complete"`` event.

        Raises:
            ImportError: If ``openai`` is not installed.
            RuntimeError: If the streaming API call fails.
        """
        import openai

        messages = self._build_messages(prompt, schema, constraints)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0,
            "stream": True,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if seed is not None:
            kwargs["seed"] = seed
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
        if stop is not None:
            kwargs["stop"] = stop
        if constraints == "json" or schema:
            kwargs["response_format"] = {"type": "json_object"}

        client = self._get_client()
        t0 = time.monotonic()
        accumulated = ""

        try:
            async with await client.chat.completions.create(**kwargs) as stream:
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
            raise RuntimeError(f"Fireworks API streaming error: {exc}") from exc

        yield StreamEvent(
            type="complete",
            content=accumulated,
            backend=self.name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
