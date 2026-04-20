"""SGLang inference backend for FormatShield."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from formatshield._retry import API_RETRY, with_retry
from formatshield.scorer.features import StreamEvent

#: Default SGLang server base URL (OpenAI-compatible API endpoint).
_DEFAULT_BASE_URL = "http://localhost:30000/v1"

#: Default model for SGLang — a commonly deployed open-weight model.
_DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


class SGLangBackend:
    """
    FormatShield backend that targets a local `SGLang <https://github.com/sgl-project/sglang>`_
    inference server via its OpenAI-compatible REST API.

    SGLang implements RadixAttention for efficient KV-cache prefix reuse, so
    :attr:`supports_kv_cache_reuse` is ``True``.  Grammar-constrained
    generation is supported via the ``ebnf`` extension body parameter.

    Parameters
    ----------
    base_url:
        Base URL of the SGLang OpenAI-compatible endpoint.
        Defaults to ``"http://localhost:30000/v1"``.
    api_key:
        Placeholder API key sent to the server.  SGLang does not enforce
        authentication for local deployments; defaults to ``"EMPTY"``.
    model:
        Model identifier.  Accepts both plain model names (e.g.
        ``"meta-llama/Llama-3.1-8B-Instruct"``) and the ``"sglang/model-name"``
        prefixed format used by FormatShield's router.
    """

    #: Backend identifier consumed by the FormatShield router.
    name: str = "sglang"

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._base_url = base_url
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
        )
        # Strip optional "sglang/" prefix so the raw model name reaches the API.
        self.model = model.removeprefix("sglang/")

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supports_kv_cache_reuse(self) -> bool:
        """SGLang supports RadixAttention-based KV-cache prefix reuse."""
        return True

    @property
    def accuracy_loss_baseline(self) -> float | None:
        """
        20 % baseline accuracy loss for structured-output generation, as
        measured across FormatShield's internal benchmark suite for
        grammar-constrained SGLang deployments.
        """
        return 0.20

    @property
    def supports_logit_bias(self) -> bool:
        """This backend does not support token-level logit biasing."""
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        schema: dict | None,  # type: ignore[type-arg]
        constraints: str | None,
    ) -> list[dict]:  # type: ignore[type-arg]
        """Assemble the OpenAI-compatible messages list.

        When a schema is provided or constraints == "json", a system message
        is prepended with JSON formatting instructions.

        Args:
            prompt: The raw user prompt.
            schema: Optional JSON schema dict.
            constraints: Optional constraint hint string (``"json"`` for JSON mode).

        Returns:
            List of message dicts suitable for the chat completions API.
        """
        messages: list[dict] = []  # type: ignore[type-arg]

        if schema:
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
        elif constraints == "json":
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Respond with valid JSON only. "
                        "Return only the JSON object with no surrounding text."
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
        schema: dict | None = None,  # type: ignore[type-arg]
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

        JSON mode is activated when *schema* is provided or *constraints* is
        ``"json"``.  Grammar constraints are passed via ``extra_body={"ebnf": ...}``
        when *constraints* is a grammar string other than ``"json"``.
        KV-cache prefix reuse is passed via ``extra_body={"kv_cache_prefix": ...}``
        when *kv_cache_prefix* is provided.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict; activates JSON-object response mode.
            constraints: Pass ``"json"`` for JSON mode, or an EBNF grammar string
                for grammar-constrained generation.
            kv_cache_prefix: Optional prefix for SGLang RadixAttention cache reuse.
            temperature: Sampling temperature.  Defaults to ``0`` for deterministic output.
            top_p: Nucleus sampling probability.  ``None`` defers to the API default.
            top_k: Top-k sampling cutoff.  ``None`` defers to the API default.
            max_tokens: Maximum number of tokens to generate.
            seed: Random seed for reproducible sampling.
            frequency_penalty: Frequency penalty (OpenAI-compatible).
            presence_penalty: Presence penalty (OpenAI-compatible).
            stop: Stop sequence(s).  ``None`` defers to the API default.

        Returns:
            The model's response text.

        Raises:
            RuntimeError: Wraps any unretried SGLang API error with a human-readable message.
        """
        messages = self._build_messages(prompt, schema, constraints)

        kwargs: dict = {  # type: ignore[type-arg]
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

        # Activate JSON-object mode when schema or "json" constraint is present.
        if schema or constraints == "json":
            kwargs["response_format"] = {"type": "json_object"}

        # Build extra_body for SGLang-specific extensions.
        extra_body: dict = {}  # type: ignore[type-arg]
        if constraints and constraints != "json":
            extra_body["ebnf"] = constraints
        if kv_cache_prefix is not None:
            extra_body["kv_cache_prefix"] = kv_cache_prefix
        if extra_body:
            kwargs["extra_body"] = extra_body

        async def _call() -> str:
            try:
                response = await self._client.chat.completions.create(**kwargs)
            except (openai.RateLimitError, openai.InternalServerError):
                raise  # propagate retryable errors un-wrapped
            except openai.APIError as exc:
                raise RuntimeError(f"SGLang API error: {exc}") from exc
            content = response.choices[0].message.content
            return content if content is not None else ""

        return await with_retry(
            _call,
            API_RETRY,
            retryable=(openai.RateLimitError, openai.InternalServerError),
            operation_name=f"sglang.generate({self.model})",
        )

    async def stream(
        self,
        prompt: str,
        schema: dict | None = None,  # type: ignore[type-arg]
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
            constraints: Pass ``"json"`` for JSON mode, or an EBNF grammar string.
            temperature: Sampling temperature.  Defaults to ``0``.
            top_p: Nucleus sampling probability.  ``None`` defers to the API default.
            top_k: Top-k sampling cutoff.  ``None`` defers to the API default.
            max_tokens: Maximum number of tokens to generate.
            seed: Random seed for reproducible sampling.
            frequency_penalty: Frequency penalty (OpenAI-compatible).
            presence_penalty: Presence penalty (OpenAI-compatible).
            stop: Stop sequence(s).  ``None`` defers to the API default.

        Yields:
            Incremental output tokens followed by a final complete event.

        Raises:
            RuntimeError: Wraps any SGLang API streaming error with a human-readable message.
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
        schema: dict | None,  # type: ignore[type-arg]
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

        kwargs: dict = {  # type: ignore[type-arg]
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

        if schema or constraints == "json":
            kwargs["response_format"] = {"type": "json_object"}

        extra_body: dict = {}  # type: ignore[type-arg]
        if constraints and constraints != "json":
            extra_body["ebnf"] = constraints
        if extra_body:
            kwargs["extra_body"] = extra_body

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
            raise RuntimeError(f"SGLang API streaming error: {exc}") from exc

        yield StreamEvent(
            type="complete",
            content=accumulated,
            backend=self.name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
