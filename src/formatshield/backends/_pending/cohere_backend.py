"""Cohere inference backend for FormatShield."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from formatshield._retry import API_RETRY, with_retry
from formatshield.scorer.features import StreamEvent


class CohereBackend:
    """
    FormatShield backend that targets the `Cohere <https://cohere.com>`_ API.

    Cohere provides high-quality command models with native JSON response
    mode.  It does **not** expose server-side KV-cache prefix reuse, so
    :attr:`supports_kv_cache_reuse` is ``False``.

    Parameters
    ----------
    api_key:
        Cohere API key.  If ``None``, the value of the ``COHERE_API_KEY``
        environment variable is used.  A :exc:`ValueError` is raised when
        neither source provides a key.
    model:
        Model identifier.  Accepts both plain model names (e.g.
        ``"command-r-plus"``) and the ``"cohere/model-name"`` prefixed
        format used by FormatShield's router.
    """

    #: Backend identifier consumed by the FormatShield router.
    name: str = "cohere"

    #: Cohere does not expose server-side KV-cache prefix reuse.
    supports_kv_cache_reuse: bool = False

    #: 12% baseline accuracy loss measured on FormatShield's benchmark suite.
    accuracy_loss_baseline: float | None = 0.12

    #: Cohere does not support token-level logit biasing.
    supports_logit_bias: bool = False

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "command-r-plus",
    ) -> None:
        resolved_key = api_key or os.environ.get("COHERE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Cohere API key supplied.  Pass api_key= or set the "
                "COHERE_API_KEY environment variable."
            )
        try:
            import cohere  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "The 'cohere' package is required for CohereBackend. "
                "Install it with: pip install 'formatshield[cohere]'"
            ) from exc
        self._client = cohere.AsyncClientV2(api_key=resolved_key)
        self._cohere = cohere
        # Strip optional "cohere/" prefix so the raw model name reaches the API.
        self.model = model.removeprefix("cohere/")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
        constraints: str | None,
    ) -> list[dict[str, Any]]:
        """Assemble the Cohere messages list."""
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
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: list[str] | str | None = None,
        logit_bias: dict[int, float] | None = None,
    ) -> str:
        """Generate a response and return the full text.

        When *schema* is provided **and** *constraints* is ``"json"``,
        Cohere's native JSON response mode is activated via
        ``response_format={"type": "json_object"}``. When only *schema* is
        provided, it is embedded in a system prompt.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict.
            constraints: Pass ``"json"`` to activate Cohere JSON-mode.
            kv_cache_prefix: Ignored; Cohere does not support prefix caching.
            temperature: Sampling temperature. Defaults to ``0``.
            top_p: Nucleus sampling probability. ``None`` defers to the API default.
            top_k: Top-k sampling cutoff. ``None`` defers to the API default.
            max_tokens: Maximum tokens to generate. ``None`` defers to the API default.
            seed: Random seed. ``None`` defers to the API default.
            frequency_penalty: Frequency penalty. ``None`` defers to the API default.
            presence_penalty: Presence penalty. ``None`` defers to the API default.
            stop: Stop sequence(s). ``None`` defers to the API default.

        Returns:
            The model's response text.

        Raises:
            RuntimeError: Wraps any Cohere API error with a human-readable message.
        """
        del kv_cache_prefix  # Cohere does not support prefix caching
        messages = self._build_messages(prompt, schema, constraints)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0,
        }

        if top_p is not None:
            kwargs["p"] = top_p
        if top_k is not None:
            kwargs["k"] = top_k
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if seed is not None:
            kwargs["seed"] = seed
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
        if stop is not None:
            kwargs["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        if constraints == "json":
            kwargs["response_format"] = {"type": "json_object"}

        async def _call() -> str:
            try:
                response = await self._client.chat(**kwargs)
            except Exception as exc:
                exc_name = type(exc).__name__
                if "RateLimitError" in exc_name or "TooManyRequests" in exc_name:
                    raise
                raise RuntimeError(f"Cohere API error: {exc}") from exc
            content = response.message.content
            if content and len(content) > 0:
                text = content[0].text
                return text if text is not None else ""
            return ""

        return await with_retry(
            _call,
            API_RETRY,
            retryable=(Exception,),
            operation_name=f"cohere.generate({self.model})",
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
        logit_bias: dict[int, float] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the model's response as StreamEvent objects.

        Yields one ``"output"`` event per incremental token chunk, then a
        single ``"complete"`` event containing the fully assembled text.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict.
            constraints: Pass ``"json"`` to activate Cohere JSON-mode.
            temperature: Sampling temperature. Defaults to ``0``.
            top_p: Nucleus sampling probability. ``None`` defers to the API default.
            top_k: Top-k sampling cutoff. ``None`` defers to the API default.
            max_tokens: Maximum tokens to generate. ``None`` defers to the API default.
            seed: Random seed. ``None`` defers to the API default.
            frequency_penalty: Frequency penalty. ``None`` defers to the API default.
            presence_penalty: Presence penalty. ``None`` defers to the API default.
            stop: Stop sequence(s). ``None`` defers to the API default.

        Yields:
            Incremental output tokens followed by a final complete event.

        Raises:
            RuntimeError: Wraps any Cohere API error with a human-readable message.
        """
        return self._stream_impl(
            prompt,
            schema,
            constraints,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )

    async def _stream_impl(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
        constraints: str | None,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: list[str] | str | None = None,
        logit_bias: dict[int, float] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        messages = self._build_messages(prompt, schema, constraints)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0,
        }

        if top_p is not None:
            kwargs["p"] = top_p
        if top_k is not None:
            kwargs["k"] = top_k
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if seed is not None:
            kwargs["seed"] = seed
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
        if stop is not None:
            kwargs["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        if constraints == "json":
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.monotonic()
        accumulated = ""

        try:
            async for event in await self._client.chat_stream(**kwargs):
                if hasattr(event, "type"):
                    if event.type == "content-delta":
                        delta = event.delta.message.content.text
                        if delta:
                            accumulated += delta
                            yield StreamEvent(
                                type="output",
                                token=delta,
                                backend=self.name,
                                latency_ms=(time.monotonic() - t0) * 1000,
                            )
        except Exception as exc:
            raise RuntimeError(f"Cohere API streaming error: {exc}") from exc

        yield StreamEvent(
            type="complete",
            content=accumulated,
            backend=self.name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
