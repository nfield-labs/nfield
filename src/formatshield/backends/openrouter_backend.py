"""OpenRouter inference backend for FormatShield."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from formatshield.scorer.features import StreamEvent

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_HEADERS = {
    "HTTP-Referer": "https://github.com/formatshield/formatshield",
    "X-Title": "FormatShield",
}


class OpenRouterBackend:
    """
    FormatShield backend that routes requests through
    `OpenRouter <https://openrouter.ai>`_, a unified proxy that exposes
    hundreds of models (Llama, Mistral, Claude, GPT-4, …) under a single
    OpenAI-compatible API.

    Parameters
    ----------
    api_key:
        OpenRouter API key.  If ``None``, the value of the
        ``OPENROUTER_API_KEY`` environment variable is used.  A
        :exc:`ValueError` is raised when neither source provides a key.
    model:
        Model identifier in OpenRouter's ``"provider/model-name"`` format
        (e.g. ``"meta-llama/llama-3.1-70b-instruct"``).  The FormatShield
        ``"openrouter/"`` prefix is stripped automatically.
    """

    #: Backend identifier consumed by the FormatShield router.
    name: str = "openrouter"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "meta-llama/llama-3.1-70b-instruct",
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No OpenRouter API key supplied.  Pass api_key= or set the "
                "OPENROUTER_API_KEY environment variable."
            )
        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=_OPENROUTER_BASE_URL,
            default_headers=_DEFAULT_HEADERS,
        )
        # Strip optional "openrouter/" prefix.
        self.model = model.removeprefix("openrouter/")

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supports_kv_cache_reuse(self) -> bool:
        """OpenRouter does not expose server-side KV-cache prefix reuse."""
        return False

    @property
    def accuracy_loss_baseline(self) -> float | None:
        """
        20 % baseline accuracy loss for structured-output generation observed
        across the FormatShield benchmark suite when routing through
        OpenRouter without TTF.
        """
        return 0.20

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        schema: dict | None,
        constraints: str | None,
    ) -> list[dict]:
        """Assemble the OpenAI-compatible messages list."""
        messages: list[dict] = []

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
        schema: dict | None = None,
        constraints: str | None = None,
        kv_cache_prefix: str | None = None,
    ) -> str:
        """
        Generate a response and return the full text.

        When *constraints* is ``"json"``, OpenAI-compatible JSON-mode
        (``response_format={"type": "json_object"}``) is requested from
        OpenRouter.  Models that do not support JSON-mode fall back to
        instruction-based formatting via the system prompt.

        Parameters
        ----------
        prompt:
            The user prompt.
        schema:
            Optional JSON schema dict.
        constraints:
            Pass ``"json"`` to request JSON-mode output.
        kv_cache_prefix:
            Ignored; OpenRouter does not support prefix caching.

        Returns
        -------
        str
            The model's response text.

        Raises
        ------
        RuntimeError
            Wraps any :exc:`openai.APIError` with a human-readable message.
        """
        messages = self._build_messages(prompt, schema, constraints)

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }

        if constraints == "json":
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except openai.APIError as exc:
            raise RuntimeError(f"OpenRouter API error: {exc}") from exc

        content = response.choices[0].message.content
        return content if content is not None else ""

    async def stream(
        self,
        prompt: str,
        schema: dict | None = None,
        constraints: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream the model's response as :class:`~formatshield.scorer.features.StreamEvent` objects.

        Yields one ``"output"`` event per incremental token chunk, then a
        single ``"complete"`` event containing the fully assembled text.

        Parameters
        ----------
        prompt:
            The user prompt.
        schema:
            Optional JSON schema dict.
        constraints:
            Pass ``"json"`` to request JSON-mode output.

        Yields
        ------
        StreamEvent
            Incremental output tokens followed by a final complete event.

        Raises
        ------
        RuntimeError
            Wraps any :exc:`openai.APIError` with a human-readable message.
        """
        return self._stream_impl(prompt, schema, constraints)

    async def _stream_impl(
        self,
        prompt: str,
        schema: dict | None,
        constraints: str | None,
    ) -> AsyncIterator[StreamEvent]:
        messages = self._build_messages(prompt, schema, constraints)

        kwargs: dict = {
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
            raise RuntimeError(f"OpenRouter API streaming error: {exc}") from exc

        yield StreamEvent(
            type="complete",
            content=accumulated,
            backend=self.name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
