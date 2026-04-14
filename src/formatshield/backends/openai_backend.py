"""OpenAI inference backend for FormatShield."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from formatshield._retry import API_RETRY, with_retry
from formatshield.scorer.features import StreamEvent


class OpenAIBackend:
    """
    FormatShield backend that targets the `OpenAI <https://openai.com>`_ hosted
    inference API.

    OpenAI provides general-purpose chat and completion models.  It does
    **not** expose server-side KV-cache prefix reuse, so
    :attr:`supports_kv_cache_reuse` is ``False``.

    Parameters
    ----------
    api_key:
        OpenAI API key.  If ``None``, the value of the ``OPENAI_API_KEY``
        environment variable is used.  A :exc:`ValueError` is raised when
        neither source provides a key.
    model:
        Model identifier.  Accepts both plain OpenAI model names (e.g.
        ``"gpt-4o-mini"``) and the ``"openai/model-name"``
        prefixed format used by FormatShield's router.
    """

    #: Backend identifier consumed by the FormatShield router.
    name: str = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No OpenAI API key supplied.  Pass api_key= or set the "
                "OPENAI_API_KEY environment variable."
            )
        self._client = AsyncOpenAI(api_key=resolved_key)
        # Strip optional "openai/" prefix so the raw model name reaches the API.
        self.model = model.removeprefix("openai/")

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supports_kv_cache_reuse(self) -> bool:
        """OpenAI does not expose server-side KV-cache prefix reuse."""
        return False

    @property
    def accuracy_loss_baseline(self) -> float | None:
        """
        15 % baseline accuracy loss for structured-output generation, as
        measured across FormatShield's internal benchmark suite and corroborated
        by the OpenAI JSON-mode evaluation literature.
        """
        return 0.15

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
            # Embed the schema description into a system message so the model
            # is aware of the expected output structure even without JSON mode.
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
        """
        Generate a response and return the full text.

        When *constraints* is ``"json"``, OpenAI's native JSON-mode
        (``response_format={"type": "json_object"}``) is activated.  When
        only *schema* is provided without the ``"json"`` constraint string,
        the schema is embedded in a system prompt so the model understands
        the expected structure.

        Parameters
        ----------
        prompt:
            The user prompt.
        schema:
            Optional JSON schema dict.
        constraints:
            Pass ``"json"`` to activate OpenAI JSON-mode.
        kv_cache_prefix:
            Ignored; OpenAI does not support prefix caching.
        temperature:
            Sampling temperature.  Defaults to ``0`` for deterministic output.
        top_p:
            Nucleus sampling probability.  ``None`` defers to the API default.
        top_k:
            Ignored; OpenAI does not expose a top-k parameter.
        max_tokens:
            Maximum number of tokens to generate.  ``None`` defers to the API
            default.
        seed:
            Random seed for reproducible sampling.  ``None`` defers to the API
            default.
        frequency_penalty:
            Frequency penalty.  ``None`` defers to the API default.
        presence_penalty:
            Presence penalty.  ``None`` defers to the API default.
        stop:
            Stop sequence(s).  ``None`` defers to the API default.

        Returns
        -------
        str
            The model's response text.

        Raises
        ------
        RuntimeError
            Wraps any :exc:`openai.APIError` with a human-readable message.
        """
        del kv_cache_prefix  # OpenAI does not support prefix caching
        messages = self._build_messages(prompt, schema, constraints)

        kwargs: dict = {
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

        # Enable OpenAI JSON mode when the "json" constraint is present.
        if constraints == "json":
            kwargs["response_format"] = {"type": "json_object"}

        async def _call() -> str:
            try:
                response = await self._client.chat.completions.create(**kwargs)
            except (
                openai.RateLimitError,
                openai.InternalServerError,
                openai.APIConnectionError,
            ):
                raise  # propagate retryable errors un-wrapped
            except openai.APIError as exc:
                raise RuntimeError(f"OpenAI API error: {exc}") from exc
            content = response.choices[0].message.content
            return content if content is not None else ""

        return await with_retry(
            _call,
            API_RETRY,
            retryable=(
                openai.RateLimitError,
                openai.InternalServerError,
                openai.APIConnectionError,
            ),
            operation_name=f"openai.generate({self.model})",
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
            Pass ``"json"`` to activate OpenAI JSON-mode.
        temperature:
            Sampling temperature.  Defaults to ``0`` for deterministic output.
        top_p:
            Nucleus sampling probability.  ``None`` defers to the API default.
        top_k:
            Ignored; OpenAI does not expose a top-k parameter.
        max_tokens:
            Maximum number of tokens to generate.  ``None`` defers to the API
            default.
        seed:
            Random seed for reproducible sampling.  ``None`` defers to the API
            default.
        frequency_penalty:
            Frequency penalty.  ``None`` defers to the API default.
        presence_penalty:
            Presence penalty.  ``None`` defers to the API default.
        stop:
            Stop sequence(s).  ``None`` defers to the API default.

        Yields
        ------
        StreamEvent
            Incremental output tokens followed by a final complete event.

        Raises
        ------
        RuntimeError
            Wraps any :exc:`openai.APIError` with a human-readable message.
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
    ) -> AsyncIterator[StreamEvent]:
        messages = self._build_messages(prompt, schema, constraints)

        kwargs: dict = {
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
            raise RuntimeError(f"OpenAI API streaming error: {exc}") from exc

        yield StreamEvent(
            type="complete",
            content=accumulated,
            backend=self.name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
