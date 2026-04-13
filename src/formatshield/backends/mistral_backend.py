"""Mistral AI inference backend for FormatShield."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from formatshield._retry import API_RETRY, with_retry
from formatshield.scorer.features import StreamEvent


class MistralBackend:
    """
    FormatShield backend that targets the `Mistral AI <https://mistral.ai>`_ API.

    Mistral AI provides high-quality Mixtral models with JSON response mode
    support.  It does **not** expose server-side KV-cache prefix reuse, so
    :attr:`supports_kv_cache_reuse` is ``False``.

    Parameters
    ----------
    api_key:
        Mistral AI API key.  If ``None``, the value of the
        ``MISTRAL_API_KEY`` environment variable is used.  A
        :exc:`ValueError` is raised when neither source provides a key.
    model:
        Model identifier.  Accepts both plain model names (e.g.
        ``"mistral-large-latest"``) and the ``"mistral/model-name"``
        prefixed format used by FormatShield's router.
    """

    #: Backend identifier consumed by the FormatShield router.
    name: str = "mistral"

    #: Mistral AI does not expose server-side KV-cache prefix reuse.
    supports_kv_cache_reuse: bool = False

    #: 14% baseline accuracy loss measured on FormatShield's benchmark suite.
    accuracy_loss_baseline: float | None = 0.14

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "mistral-large-latest",
    ) -> None:
        resolved_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Mistral AI API key supplied.  Pass api_key= or set the "
                "MISTRAL_API_KEY environment variable."
            )
        try:
            from mistralai import Mistral  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "The 'mistralai' package is required for MistralBackend. "
                "Install it with: pip install 'formatshield[mistral]'"
            ) from exc
        self._client = Mistral(api_key=resolved_key)
        # Strip optional "mistral/" prefix so the raw model name reaches the API.
        self.model = model.removeprefix("mistral/")

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
        Mistral's native JSON response mode is activated via
        ``response_format={"type": "json_object"}``. When only *schema* is
        provided, it is embedded in a system prompt.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict.
            constraints: Pass ``"json"`` to activate Mistral JSON-mode.
            kv_cache_prefix: Ignored; Mistral AI does not support prefix caching.

        Returns:
            The model's response text.

        Raises:
            RuntimeError: Wraps any Mistral API error with a human-readable message.
        """
        del kv_cache_prefix  # Mistral AI does not support prefix caching
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
                response = await self._client.chat.complete_async(**kwargs)
            except Exception as exc:
                exc_name = type(exc).__name__
                if "RateLimit" in exc_name or "TooManyRequests" in exc_name:
                    raise
                raise RuntimeError(f"Mistral API error: {exc}") from exc
            if response.choices and response.choices[0].message.content:
                return str(response.choices[0].message.content)
            return ""

        return await with_retry(
            _call,
            API_RETRY,
            retryable=(Exception,),
            operation_name=f"mistral.generate({self.model})",
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
            constraints: Pass ``"json"`` to activate Mistral JSON-mode.

        Yields:
            Incremental output tokens followed by a final complete event.

        Raises:
            RuntimeError: Wraps any Mistral API error with a human-readable message.
        """
        return self._stream_impl(prompt, schema, constraints)

    async def _stream_impl(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
        constraints: str | None,
    ) -> AsyncIterator[StreamEvent]:
        messages = self._build_messages(prompt, schema, constraints)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }

        if constraints == "json":
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.monotonic()
        accumulated = ""

        try:
            async for chunk in await self._client.chat.stream_async(**kwargs):
                delta = chunk.data.choices[0].delta.content if chunk.data.choices else None
                if delta:
                    accumulated += delta
                    yield StreamEvent(
                        type="output",
                        token=delta,
                        backend=self.name,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
        except Exception as exc:
            raise RuntimeError(f"Mistral API streaming error: {exc}") from exc

        yield StreamEvent(
            type="complete",
            content=accumulated,
            backend=self.name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
