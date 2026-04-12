"""Ollama local inference backend for FormatShield."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from ollama import AsyncClient, ResponseError

from formatshield.scorer.features import StreamEvent


class OllamaBackend:
    """
    FormatShield backend that targets a locally-running
    `Ollama <https://ollama.com>`_ server.

    Ollama manages model weights on the local machine and exposes a simple
    REST API.  It supports native JSON-mode output via the ``format="json"``
    option.

    Parameters
    ----------
    host:
        Base URL of the Ollama server.  Defaults to
        ``"http://localhost:11434"``.
    model:
        Model tag to use (e.g. ``"llama3.1:70b"``).  The FormatShield
        ``"ollama/"`` prefix is stripped automatically.
    """

    #: Backend identifier consumed by the FormatShield router.
    name: str = "ollama"

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.1:70b",
    ) -> None:
        self.host = host
        # Strip optional "ollama/" prefix so the raw tag reaches the API.
        self.model = model.removeprefix("ollama/")

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supports_kv_cache_reuse(self) -> bool:
        """Ollama does not expose server-side KV-cache prefix reuse."""
        return False

    @property
    def accuracy_loss_baseline(self) -> float | None:
        """
        22 % baseline accuracy loss for structured-output generation on
        locally-hosted Ollama models, as measured across the FormatShield
        benchmark suite.  The higher value compared to hosted APIs reflects
        the wider variety of quantisation levels and model versions typically
        used in self-hosted setups.
        """
        return 0.22

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        schema: dict | None,
        constraints: str | None,
    ) -> list[dict]:
        """Assemble the Ollama-compatible messages list."""
        import json

        messages: list[dict] = []

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
        Generate a response using the local Ollama server and return the full
        text.

        When *schema* is provided, Ollama's ``format="json"`` mode is enabled,
        which constrains the model's sampler to produce only valid JSON tokens.

        Parameters
        ----------
        prompt:
            The user prompt.
        schema:
            Optional JSON schema dict.  When present, JSON-mode is activated.
        constraints:
            Optional constraint hint string.  Currently not used beyond the
            *schema* check.
        kv_cache_prefix:
            Ignored; Ollama does not support prefix caching.

        Returns
        -------
        str
            The model's response text (``message.content``).

        Raises
        ------
        RuntimeError
            Wraps :exc:`ollama.ResponseError` with a human-readable message
            that includes advice for missing models.
        """
        client = AsyncClient(host=self.host)
        messages = self._build_messages(prompt, schema, constraints)

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": 0},
        }

        if schema:
            kwargs["format"] = "json"

        try:
            response = await client.chat(**kwargs)
        except ResponseError as exc:
            model_hint = ""
            if "not found" in str(exc).lower() or "does not exist" in str(exc).lower():
                model_hint = f"  Hint: run `ollama pull {self.model}` to download the model."
            raise RuntimeError(
                f"Ollama error for model '{self.model}': {exc}.{model_hint}"
            ) from exc

        return response.message.content or ""

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
            Optional JSON schema dict.  When present, JSON-mode is activated.
        constraints:
            Optional constraint hint string.

        Yields
        ------
        StreamEvent
            Incremental output tokens followed by a final complete event.

        Raises
        ------
        RuntimeError
            Wraps :exc:`ollama.ResponseError` with a human-readable message.
        """
        return self._stream_impl(prompt, schema, constraints)

    async def _stream_impl(
        self,
        prompt: str,
        schema: dict | None,
        constraints: str | None,
    ) -> AsyncIterator[StreamEvent]:
        client = AsyncClient(host=self.host)
        messages = self._build_messages(prompt, schema, constraints)

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0},
        }

        if schema:
            kwargs["format"] = "json"

        t0 = time.monotonic()
        accumulated = ""

        try:
            async for chunk in await client.chat(**kwargs):
                token = chunk.message.content
                if token:
                    accumulated += token
                    yield StreamEvent(
                        type="output",
                        token=token,
                        backend=self.name,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
        except ResponseError as exc:
            model_hint = ""
            if "not found" in str(exc).lower() or "does not exist" in str(exc).lower():
                model_hint = f"  Hint: run `ollama pull {self.model}` to download the model."
            raise RuntimeError(
                f"Ollama streaming error for model '{self.model}': {exc}.{model_hint}"
            ) from exc

        yield StreamEvent(
            type="complete",
            content=accumulated,
            backend=self.name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
