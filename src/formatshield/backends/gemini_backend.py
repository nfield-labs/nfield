"""Google Gemini inference backend for FormatShield."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator

from formatshield._retry import API_RETRY, with_retry
from formatshield.scorer.features import StreamEvent


class GeminiBackend:
    """
    FormatShield backend that targets the `Google Gemini <https://ai.google.dev>`_
    hosted inference API.

    Gemini provides multimodal language models from Google DeepMind.  It does
    **not** expose server-side KV-cache prefix reuse, so
    :attr:`supports_kv_cache_reuse` is ``False``.

    ``google-generativeai`` is an *optional* dependency.  It is imported lazily
    inside each method so that the rest of FormatShield remains importable even
    when the library is not installed.

    Parameters
    ----------
    api_key:
        Google AI API key.  If ``None``, the value of the ``GEMINI_API_KEY``
        environment variable is used.  A :exc:`ValueError` is raised when
        neither source provides a key.
    model:
        Model identifier.  Accepts both plain Gemini model names (e.g.
        ``"gemini-2.0-flash"``) and the ``"gemini/model-name"``
        prefixed format used by FormatShield's router.
    """

    #: Backend identifier consumed by the FormatShield router.
    name: str = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
    ) -> None:
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Gemini API key supplied.  Pass api_key= or set the "
                "GEMINI_API_KEY environment variable."
            )
        self._api_key = resolved_key
        # Strip optional "gemini/" prefix so the raw model name reaches the API.
        self._model_name = model.removeprefix("gemini/")

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supports_kv_cache_reuse(self) -> bool:
        """Gemini does not expose server-side KV-cache prefix reuse."""
        return False

    @property
    def accuracy_loss_baseline(self) -> float | None:
        """
        14 % baseline accuracy loss for structured-output generation, as
        measured across FormatShield's internal benchmark suite for
        Gemini constrained-decoding modes.
        """
        return 0.14

    @property
    def supports_logit_bias(self) -> bool:
        """This backend does not support token-level logit biasing."""
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, prompt: str, schema: dict | None) -> str:  # type: ignore[type-arg]
        """Build the full prompt string, embedding schema instructions when provided.

        Args:
            prompt: The raw user prompt.
            schema: Optional JSON schema dict to embed as a system instruction.

        Returns:
            Full prompt string to send to the Gemini model.
        """
        if schema:
            schema_text = json.dumps(schema, indent=2)
            system_instruction = (
                "You must respond with valid JSON that conforms to the "
                f"following JSON schema:\n\n{schema_text}\n\n"
                "Do not include any text outside the JSON object.\n\n"
            )
            return system_instruction + prompt
        return prompt

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

        When *schema* is provided, JSON mode is activated via
        ``response_mime_type="application/json"`` and the schema is also
        embedded in the prompt.  The *seed*, *frequency_penalty*, and
        *presence_penalty* parameters are silently ignored as Gemini does
        not support them.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict.
            constraints: Ignored; Gemini JSON mode is activated when schema is provided.
            kv_cache_prefix: Ignored; Gemini does not support prefix caching.
            temperature: Sampling temperature.  Defaults to ``0.0`` for deterministic output.
            top_p: Nucleus sampling probability.  ``None`` defers to the API default.
            top_k: Top-k sampling cutoff.  ``None`` defers to the API default.
            max_tokens: Maximum number of tokens to generate (``max_output_tokens``).
            seed: Ignored; Gemini does not support a random seed parameter.
            frequency_penalty: Ignored; Gemini does not support frequency penalty.
            presence_penalty: Ignored; Gemini does not support presence penalty.
            stop: Stop sequence(s).  ``None`` defers to the API default.

        Returns:
            The model's response text.

        Raises:
            ImportError: If ``google-generativeai`` is not installed.
            RuntimeError: Wraps any unretried Gemini API error with a human-readable message.
        """
        try:
            import google.api_core.exceptions  # pyright: ignore[reportMissingImports]
            import google.generativeai as genai  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is required for GeminiBackend.  "
                "Install it with: pip install google-generativeai"
            ) from exc

        # Accepted for protocol compatibility — Gemini does not support these params.
        del constraints, kv_cache_prefix, seed, frequency_penalty, presence_penalty

        full_prompt = self._build_prompt(prompt, schema)

        stop_sequences: list[str] | None = None
        if stop is not None:
            stop_sequences = stop if isinstance(stop, list) else [stop]

        async def _call() -> str:
            try:
                genai.configure(api_key=self._api_key)
                model_obj = genai.GenerativeModel(self._model_name)
                generation_config = genai.GenerationConfig(
                    response_mime_type="application/json" if schema else None,
                    temperature=temperature if temperature is not None else 0.0,
                    max_output_tokens=max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    stop_sequences=stop_sequences,
                )
                response = await model_obj.generate_content_async(
                    full_prompt,
                    generation_config=generation_config,
                )
            except (
                google.api_core.exceptions.ResourceExhausted,
                google.api_core.exceptions.ServiceUnavailable,
            ):
                raise  # propagate retryable errors un-wrapped
            except google.api_core.exceptions.GoogleAPIError as exc:
                raise RuntimeError(f"Gemini API error: {exc}") from exc
            return response.text if response.text is not None else ""

        return await with_retry(
            _call,
            API_RETRY,
            retryable=(
                google.api_core.exceptions.ResourceExhausted,
                google.api_core.exceptions.ServiceUnavailable,
            ),
            operation_name=f"gemini.generate({self._model_name})",
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
        The *seed*, *frequency_penalty*, and *presence_penalty* parameters
        are silently ignored.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict.
            constraints: Ignored; JSON mode is activated when schema is provided.
            temperature: Sampling temperature.  Defaults to ``0.0``.
            top_p: Nucleus sampling probability.  ``None`` defers to the API default.
            top_k: Top-k sampling cutoff.  ``None`` defers to the API default.
            max_tokens: Maximum number of tokens to generate.
            seed: Ignored; Gemini does not support a random seed parameter.
            frequency_penalty: Ignored; Gemini does not support frequency penalty.
            presence_penalty: Ignored; Gemini does not support presence penalty.
            stop: Stop sequence(s).  ``None`` defers to the API default.

        Yields:
            Incremental output tokens followed by a final complete event.

        Raises:
            ImportError: If ``google-generativeai`` is not installed.
            RuntimeError: Wraps any Gemini API streaming error with a human-readable message.
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
        try:
            import google.api_core.exceptions  # pyright: ignore[reportMissingImports]
            import google.generativeai as genai  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is required for GeminiBackend.  "
                "Install it with: pip install google-generativeai"
            ) from exc

        # Accepted for protocol compatibility — Gemini does not support these params.
        del constraints, seed, frequency_penalty, presence_penalty

        full_prompt = self._build_prompt(prompt, schema)

        stop_sequences: list[str] | None = None
        if stop is not None:
            stop_sequences = stop if isinstance(stop, list) else [stop]

        t0 = time.monotonic()
        accumulated = ""

        try:
            genai.configure(api_key=self._api_key)
            model_obj = genai.GenerativeModel(self._model_name)
            generation_config = genai.GenerationConfig(
                response_mime_type="application/json" if schema else None,
                temperature=temperature if temperature is not None else 0.0,
                max_output_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                stop_sequences=stop_sequences,
            )
            async for chunk in await model_obj.generate_content_async(
                full_prompt,
                generation_config=generation_config,
                stream=True,
            ):
                token = chunk.text if chunk.text is not None else ""
                if token:
                    accumulated += token
                    yield StreamEvent(
                        type="output",
                        token=token,
                        backend=self.name,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
        except google.api_core.exceptions.GoogleAPIError as exc:
            raise RuntimeError(f"Gemini API streaming error: {exc}") from exc

        yield StreamEvent(
            type="complete",
            content=accumulated,
            backend=self.name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
