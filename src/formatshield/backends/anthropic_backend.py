"""Anthropic inference backend for FormatShield."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator

import anthropic  # type: ignore[import-not-found]
from anthropic import AsyncAnthropic  # type: ignore[import-not-found]

from formatshield._retry import API_RETRY, with_retry
from formatshield.scorer.features import StreamEvent


class AnthropicBackend:
    """
    FormatShield backend that targets the Anthropic hosted inference API.

    **Prompt caching:** When *kv_cache_prefix* is supplied to
    :meth:`generate`, the prefix is sent as a separate content block tagged
    with ``"cache_control": {"type": "ephemeral"}`` so that KV activations
    for the static portion of the prompt are reused across requests.
    :attr:`supports_kv_cache_reuse` is therefore ``True``.

    Parameters
    ----------
    api_key:
        Anthropic API key.  If ``None``, the value of the
        ``ANTHROPIC_API_KEY`` environment variable is used.  A
        :exc:`ValueError` is raised when neither source provides a key.
    model:
        Model identifier.  Accepts both plain Anthropic model names (e.g.
        ``"claude-3-5-haiku-20241022"``) and the ``"anthropic/model-name"``
        prefixed format used by FormatShield's router.
    """

    #: Backend identifier consumed by the FormatShield router.
    name: str = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-haiku-20241022",
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Anthropic API key supplied.  Pass api_key= or set the "
                "ANTHROPIC_API_KEY environment variable."
            )
        self._client = AsyncAnthropic(api_key=resolved_key)
        # Strip optional "anthropic/" prefix so the raw model name reaches the API.
        self.model = model.removeprefix("anthropic/")

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supports_kv_cache_reuse(self) -> bool:
        """Supports prompt caching via ``cache_control`` blocks.

        When ``kv_cache_prefix`` is supplied to :meth:`generate`, it is sent
        as a ``cache_control: {"type": "ephemeral"}`` content block so that
        KV activations for the static portion of the prompt are reused across
        requests sharing the same prefix.
        """
        return True

    @property
    def accuracy_loss_baseline(self) -> float | None:
        """Estimated accuracy loss for structured-output generation."""
        return 0.12

    @property
    def supports_logit_bias(self) -> bool:
        """This backend does not support token-level logit biasing."""
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        schema: dict | None,
        constraints: str | None,
    ) -> str | None:
        """
        Build a system prompt for JSON-constrained generation.

        Returns ``None`` when no schema or JSON constraint is active.
        """
        if constraints == "json":
            if schema:
                schema_text = json.dumps(schema, indent=2)
                return (
                    "You must respond with valid JSON that conforms to the "
                    f"following JSON schema:\n\n{schema_text}\n\n"
                    "Output only the JSON object with no surrounding text."
                )
            return (
                "You must respond with valid JSON.  "
                "Output only the JSON object with no surrounding text."
            )

        if schema:
            schema_text = json.dumps(schema, indent=2)
            return (
                "You must respond with valid JSON that conforms to the "
                f"following JSON schema:\n\n{schema_text}\n\n"
                "Do not include any text outside the JSON object."
            )

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        schema: dict | None = None,
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
        """
        Generate a response and return the full text.

        When *constraints* is ``"json"`` or a *schema* is provided, a system
        prompt instructing the model to produce valid JSON is injected
        automatically.

        Parameters
        ----------
        prompt:
            The user prompt.
        schema:
            Optional JSON schema dict.
        constraints:
            Pass ``"json"`` to request a JSON-formatted response.
        kv_cache_prefix:
            Ignored; Anthropic does not support prefix caching.
        temperature:
            Sampling temperature.  ``None`` defers to the API default.
        top_p:
            Nucleus sampling probability.  ``None`` defers to the API default.
        top_k:
            Top-k sampling cutoff.  ``None`` defers to the API default.
        max_tokens:
            Maximum number of tokens to generate.  Overrides the default of
            ``4096`` when provided.
        seed:
            Ignored; Anthropic does not expose a seed parameter.
        frequency_penalty:
            Ignored; Anthropic does not expose a frequency penalty parameter.
        presence_penalty:
            Ignored; Anthropic does not expose a presence penalty parameter.
        stop:
            Stop sequence(s).  ``None`` defers to the API default.

        Returns
        -------
        str
            The model's response text.

        Raises
        ------
        RuntimeError
            Wraps any :exc:`anthropic.APIError` with a human-readable message.
        """
        system_prompt = self._build_system_prompt(schema, constraints)

        # When a static cache prefix is supplied, structure the message as two
        # content blocks: the prefix tagged with cache_control (so KV activations
        # are reused across requests) followed by the variable prompt text.
        if kv_cache_prefix:
            message_content: list[dict] | str = [
                {
                    "type": "text",
                    "text": kv_cache_prefix,
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ]
        else:
            message_content = prompt

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens if max_tokens is not None else 4096,
            "messages": [{"role": "user", "content": message_content}],
        }

        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        if stop is not None:
            kwargs["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        if system_prompt is not None:
            kwargs["system"] = system_prompt

        async def _call() -> str:
            try:
                response = await self._client.messages.create(**kwargs)
            except (
                anthropic.RateLimitError,
                anthropic.InternalServerError,
                anthropic.APIConnectionError,
            ):
                raise  # propagate retryable errors un-wrapped
            except anthropic.APIError as exc:
                raise RuntimeError(f"Anthropic API error: {exc}") from exc
            return response.content[0].text

        return await with_retry(
            _call,
            API_RETRY,
            retryable=(
                anthropic.RateLimitError,
                anthropic.InternalServerError,
                anthropic.APIConnectionError,
            ),
            operation_name=f"anthropic.generate({self.model})",
        )

    async def stream(
        self,
        prompt: str,
        schema: dict | None = None,
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
        """
        Stream the model's response as
        :class:`~formatshield.scorer.features.StreamEvent` objects.

        Yields one ``"output"`` event per incremental token chunk, then a
        single ``"complete"`` event containing the fully assembled text.

        Parameters
        ----------
        prompt:
            The user prompt.
        schema:
            Optional JSON schema dict.
        constraints:
            Pass ``"json"`` to request a JSON-formatted response.
        temperature:
            Sampling temperature.  ``None`` defers to the API default.
        top_p:
            Nucleus sampling probability.  ``None`` defers to the API default.
        top_k:
            Top-k sampling cutoff.  ``None`` defers to the API default.
        max_tokens:
            Maximum number of tokens to generate.  Overrides the default of
            ``4096`` when provided.
        seed:
            Ignored; Anthropic does not expose a seed parameter.
        frequency_penalty:
            Ignored; Anthropic does not expose a frequency penalty parameter.
        presence_penalty:
            Ignored; Anthropic does not expose a presence penalty parameter.
        stop:
            Stop sequence(s).  ``None`` defers to the API default.

        Yields
        ------
        StreamEvent
            Incremental output tokens followed by a final complete event.

        Raises
        ------
        RuntimeError
            Wraps any :exc:`anthropic.APIError` with a human-readable message.
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
        schema: dict | None,
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
        system_prompt = self._build_system_prompt(schema, constraints)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens if max_tokens is not None else 4096,
            "messages": [{"role": "user", "content": prompt}],
        }

        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        if stop is not None:
            kwargs["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        if system_prompt is not None:
            kwargs["system"] = system_prompt

        t0 = time.monotonic()
        accumulated = ""

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for delta in stream.text_stream:
                    if delta:
                        accumulated += delta
                        yield StreamEvent(
                            type="output",
                            token=delta,
                            backend=self.name,
                            latency_ms=(time.monotonic() - t0) * 1000,
                        )
        except anthropic.APIError as exc:
            raise RuntimeError(f"Anthropic API streaming error: {exc}") from exc

        yield StreamEvent(
            type="complete",
            content=accumulated,
            backend=self.name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
