"""vLLM inference backend for FormatShield."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from formatshield._retry import API_RETRY, with_retry
from formatshield.scorer.features import StreamEvent


class VLLMBackend:
    """
    FormatShield backend that targets a self-hosted
    `vLLM <https://docs.vllm.ai>`_ server.

    vLLM is unique among FormatShield backends in that it supports
    **server-side KV-cache prefix reuse** via its automatic prefix-caching
    feature.  When a non-empty *kv_cache_prefix* is passed to
    :meth:`generate`, it is injected as an initial system message so that
    vLLM can reuse the prefix's KV activations across requests that share
    the same system prompt.

    vLLM exposes an OpenAI-compatible REST API, so this backend uses the
    official ``openai`` Python client pointed at the local server.

    Parameters
    ----------
    base_url:
        Base URL of the vLLM OpenAI-compatible server.
        Defaults to ``"http://localhost:8000/v1"``.
    api_key:
        Placeholder API key.  vLLM requires a non-empty value but does not
        validate it; defaults to ``"EMPTY"``.
    model:
        Model identifier as registered with the vLLM server (e.g.
        ``"meta-llama/Llama-3-70b-Instruct"``).  The FormatShield
        ``"vllm/"`` prefix is stripped automatically.
    """

    #: Backend identifier consumed by the FormatShield router.
    name: str = "vllm"

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "meta-llama/Llama-3-70b-Instruct",
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url
        # Strip optional "vllm/" prefix so the raw model name reaches the API.
        self.model = model.removeprefix("vllm/")

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supports_kv_cache_reuse(self) -> bool:
        """
        vLLM supports automatic server-side KV-cache prefix reuse when
        prefix caching is enabled (``--enable-prefix-caching`` flag at server
        startup).

        When :attr:`supports_kv_cache_reuse` is ``True``, the FormatShield
        router may pass a shared *kv_cache_prefix* to :meth:`generate` so
        that the prefix's KV activations are amortised across many requests.
        """
        return True

    @property
    def accuracy_loss_baseline(self) -> float | None:
        """
        23 % baseline accuracy loss for structured-output generation on
        vLLM-hosted models, as measured across the FormatShield benchmark
        suite.  The slightly higher value compared to hosted APIs reflects
        the wider variety of quantisation levels and sampling settings
        typically used in self-hosted deployments.
        """
        return 0.23

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        schema: dict | None,
        constraints: str | None,
        kv_cache_prefix: str | None,
    ) -> list[dict]:
        """Assemble the OpenAI-compatible messages list."""
        messages: list[dict] = []

        # Inject the KV-cache prefix as the first system message so vLLM can
        # detect and reuse the prefix's cached KV activations.
        if kv_cache_prefix:
            messages.append({"role": "system", "content": kv_cache_prefix})

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
        Generate a response using the local vLLM server and return the full
        text.

        When *constraints* is ``"json"``, vLLM's JSON-mode
        (``response_format={"type": "json_object"}``) is requested.  When a
        *kv_cache_prefix* is supplied it is placed as the first system message
        so that vLLM's automatic prefix-caching can amortise the encoding cost
        across requests sharing the same prefix.

        Parameters
        ----------
        prompt:
            The user prompt.
        schema:
            Optional JSON schema dict.
        constraints:
            Pass ``"json"`` to request JSON-mode output.
        kv_cache_prefix:
            Optional prefix string injected as a system message to enable
            vLLM prefix-caching.

        Returns
        -------
        str
            The model's response text.

        Raises
        ------
        RuntimeError
            Wraps any :exc:`openai.APIError` with a human-readable message.
        """
        messages = self._build_messages(prompt, schema, constraints, kv_cache_prefix)

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }

        if constraints == "json":
            kwargs["response_format"] = {"type": "json_object"}

        async def _call() -> str:
            try:
                response = await self._client.chat.completions.create(**kwargs)
            except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError):
                raise  # propagate retryable errors un-wrapped
            except openai.APIError as exc:
                raise RuntimeError(f"vLLM API error: {exc}") from exc
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
            operation_name=f"vllm.generate({self.model})",
        )

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
        # Note: kv_cache_prefix is not forwarded to streaming; prefix caching
        # primarily benefits single-shot generation where many requests share
        # the same prefix simultaneously.
        messages = self._build_messages(prompt, schema, constraints, kv_cache_prefix=None)

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
            raise RuntimeError(f"vLLM API streaming error: {exc}") from exc

        yield StreamEvent(
            type="complete",
            content=accumulated,
            backend=self.name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
