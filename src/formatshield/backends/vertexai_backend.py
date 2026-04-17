"""Google Vertex AI inference backend for FormatShield."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from typing import Any

from formatshield.scorer.features import StreamEvent


class VertexAIBackend:
    """Google Vertex AI inference backend using the Generative Models API.

    Uses the ``vertexai`` SDK (``google-cloud-aiplatform``) with Application
    Default Credentials (ADC) or an explicit service account key file set via
    the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable.

    The synchronous Vertex AI SDK calls are dispatched to a thread-pool
    executor so that the async event loop is never blocked.

    Args:
        model: Vertex AI model name.  Accepts both plain names (e.g.
            ``"gemini-2.0-flash-001"``) and the ``"vertexai/<model>"``
            prefixed format used by the FormatShield router.
        project: Google Cloud project ID.  Defaults to the
            ``GOOGLE_CLOUD_PROJECT`` environment variable.
        location: Google Cloud region.  Defaults to ``"us-central1"``.
    """

    #: Backend identifier consumed by the FormatShield router.
    name: str = "vertexai"

    def __init__(
        self,
        model: str = "gemini-2.0-flash-001",
        project: str | None = None,
        location: str = "us-central1",
    ) -> None:
        self._model_name = model.removeprefix("vertexai/")
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._location = location
        self._client: Any = None

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supports_kv_cache_reuse(self) -> bool:
        """Vertex AI does not expose server-side KV-cache prefix reuse."""
        return False

    @property
    def accuracy_loss_baseline(self) -> float | None:
        """14 % baseline accuracy loss for structured-output generation on Vertex AI."""
        return 0.14

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> Any:
        """Return (and lazily initialise) the Vertex AI GenerativeModel.

        Raises:
            ImportError: If ``google-cloud-aiplatform`` is not installed.
        """
        if self._client is None:
            try:
                import vertexai  # pyright: ignore[reportMissingImports]
                from vertexai.generative_models import (  # pyright: ignore[reportMissingImports]
                    GenerativeModel,
                )
            except ImportError as exc:
                raise ImportError(
                    "google-cloud-aiplatform is required for VertexAIBackend. "
                    "Install with: pip install 'formatshield[vertexai]'"
                ) from exc
            vertexai.init(project=self._project, location=self._location)
            self._client = GenerativeModel(self._model_name)
        return self._client

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
        """Generate a response via the Vertex AI Generative Models API.

        When a JSON schema is provided, ``response_mime_type`` is set to
        ``"application/json"`` in the generation config and the schema is also
        embedded in the prompt for models that support it.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict.  When supplied, JSON mode is
                activated and the schema is embedded in the prompt.
            constraints: Reserved for future grammar-constraint support;
                currently unused.
            kv_cache_prefix: Ignored; Vertex AI does not support prefix
                caching at the API level.
            temperature: Sampling temperature.  Defaults to ``0.0``.
            top_p: Nucleus sampling probability.  ``None`` defers to the API
                default.
            top_k: Top-k sampling.  ``None`` defers to the API default.
            max_tokens: Maximum output tokens.  ``None`` defers to the API
                default.
            seed: Ignored; not exposed by the Generative Models API.
            frequency_penalty: Ignored; not supported by Vertex AI.
            presence_penalty: Ignored; not supported by Vertex AI.
            stop: Ignored; stop sequences are not exposed in this adapter.

        Returns:
            The model's response as a plain string.

        Raises:
            ImportError: If ``google-cloud-aiplatform`` is not installed.
            RuntimeError: If the Vertex AI API call fails.
        """
        full_prompt = prompt
        if schema:
            schema_text = json.dumps(schema, indent=2)
            full_prompt = (
                f"You must respond with valid JSON conforming to:\n{schema_text}\n\n{prompt}"
            )

        def _run() -> str:
            try:
                from vertexai.generative_models import (  # pyright: ignore[reportMissingImports]
                    GenerationConfig,
                )
            except ImportError as exc:
                raise ImportError("google-cloud-aiplatform required") from exc
            model = self._get_model()
            gen_config_kwargs: dict[str, Any] = {
                "temperature": temperature if temperature is not None else 0.0,
            }
            if max_tokens is not None:
                gen_config_kwargs["max_output_tokens"] = max_tokens
            if top_p is not None:
                gen_config_kwargs["top_p"] = top_p
            if top_k is not None:
                gen_config_kwargs["top_k"] = top_k
            if schema:
                gen_config_kwargs["response_mime_type"] = "application/json"
            gen_config = GenerationConfig(**gen_config_kwargs)
            try:
                response = model.generate_content(full_prompt, generation_config=gen_config)
            except Exception as exc:
                raise RuntimeError(f"Vertex AI API error: {exc}") from exc
            return str(response.text)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run)

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

        Falls back to a single ``complete`` event wrapping the full response
        because the Vertex AI SDK's streaming interface requires a separate
        synchronous executor integration that is out of scope for this adapter.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict.
            constraints: Reserved for future use; currently unused.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling.
            max_tokens: Maximum output tokens.
            seed: Ignored.
            frequency_penalty: Ignored.
            presence_penalty: Ignored.
            stop: Ignored.

        Yields:
            A single :class:`StreamEvent` of type ``"complete"``.
        """
        result = await self.generate(
            prompt,
            schema,
            constraints,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed,
        )
        yield StreamEvent(type="complete", content=result, backend=self.name, latency_ms=0.0)
