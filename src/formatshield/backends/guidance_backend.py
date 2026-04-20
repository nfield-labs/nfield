"""Guidance structured-generation backend for FormatShield."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from formatshield.scorer.features import StreamEvent


class GuidanceBackend:
    """
    FormatShield backend that wraps the `guidance
    <https://github.com/guidance-ai/guidance>`_ library for structured
    generation.

    Guidance uses a *program* abstraction: a language model is extended with
    a mini-language of interleaved text and generation directives.  When a
    JSON schema is provided, ``guidance.json()`` injects the schema as a
    hard constraint into the program, forcing all generated content to be
    valid JSON conforming to the schema.

    Guidance introduces a small but non-zero accuracy impact compared to
    outlines because its constraint engine operates at the grammar level and
    may occasionally restrict valid continuations that are not covered by the
    grammar's finite-state representation of the schema.

    ``guidance`` is an *optional* dependency.  It is imported lazily inside
    each method so that the rest of FormatShield remains importable even when
    the library is not installed.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (e.g. ``"gpt2"``).
    backend_type:
        Guidance backend type string.  Currently ``"transformers"`` is the
        supported value; future versions may add ``"llamacpp"`` or
        ``"openai"``.
    """

    #: Backend identifier consumed by the FormatShield router.
    name: str = "guidance"

    def __init__(
        self,
        model_name: str = "gpt2",
        backend_type: str = "transformers",
    ) -> None:
        self.model_name = model_name
        self.backend_type = backend_type

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supports_kv_cache_reuse(self) -> bool:
        """
        Guidance does not expose server-side KV-cache prefix reuse.

        Guidance runs generation in-process and does not maintain a separate
        caching server, so there is no mechanism for sharing KV-cache tensors
        across distinct requests.
        """
        return False

    @property
    def accuracy_loss_baseline(self) -> float:
        """
        5 % baseline accuracy loss for structured-output generation via
        guidance.

        Guidance's grammar-based constraint engine is highly accurate but
        introduces a minimal, non-zero accuracy impact because its
        finite-state representation of a JSON schema may occasionally restrict
        continuations that are semantically valid but syntactically ambiguous
        within the grammar.
        """
        return 0.05

    @property
    def supports_logit_bias(self) -> bool:
        """This backend does not support token-level logit biasing."""
        return False

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
        """
        Generate a response using guidance structured generation and return
        the full text.

        When *schema* is provided the generation is constrained to the
        supplied JSON schema via ``guidance.json()``.  Without a schema,
        the prompt is sent as free text and the model generates
        unconstrained output.

        Parameters
        ----------
        prompt:
            The user prompt passed to the guidance program.
        schema:
            Optional JSON schema dict.  When present, guidance constrains
            generation to produce only schema-valid JSON.
        constraints:
            Optional constraint hint string.  Not used by this backend
            beyond the *schema* check; reserved for router compatibility.
        kv_cache_prefix:
            Ignored; guidance does not support prefix caching.

        Returns
        -------
        str
            The model's response text extracted from the guidance program
            state.

        Raises
        ------
        ImportError
            When the ``guidance`` package is not installed.  Install it with
            ``pip install guidance``.
        RuntimeError
            Wraps any generation-time exception with a human-readable
            message.
        """
        try:
            import guidance  # type: ignore[import-untyped]
            import guidance.models  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'guidance' package is required for GuidanceBackend. "
                "Install it with: pip install guidance"
            ) from exc

        try:
            lm = guidance.models.Transformers(self.model_name)
            if schema is not None:
                result_program = lm + prompt + guidance.json(schema=schema)
            else:
                result_program = lm + prompt
            result: str = str(result_program)
        except Exception as exc:
            raise RuntimeError(
                f"GuidanceBackend generation error for model '{self.model_name}': {exc}"
            ) from exc

        return result

    async def stream(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream the model's response as
        :class:`~formatshield.scorer.features.StreamEvent` objects.

        Guidance does not expose a native streaming API; this method
        simulates streaming by generating the complete text first and then
        yielding one ``"output"`` event per whitespace-delimited word,
        followed by a single ``"complete"`` event containing the full text.

        Parameters
        ----------
        prompt:
            The user prompt.
        schema:
            Optional JSON schema dict.  When present, guidance constrains
            generation to produce only schema-valid JSON.
        constraints:
            Optional constraint hint string.  Not used beyond the *schema*
            check; reserved for router compatibility.

        Yields
        ------
        StreamEvent
            One ``"output"`` event per word token, then a ``"complete"``
            event with the full generated text in ``content``.

        Raises
        ------
        ImportError
            When the ``guidance`` package is not installed.  Install it with
            ``pip install guidance``.
        RuntimeError
            Wraps any generation-time exception with a human-readable
            message.
        """
        t0 = time.monotonic()

        try:
            import guidance  # type: ignore[import-untyped]
            import guidance.models  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'guidance' package is required for GuidanceBackend. "
                "Install it with: pip install guidance"
            ) from exc

        try:
            lm = guidance.models.Transformers(self.model_name)
            if schema is not None:
                result_program = lm + prompt + guidance.json(schema=schema)
            else:
                result_program = lm + prompt
            full_text: str = str(result_program)
        except Exception as exc:
            raise RuntimeError(
                f"GuidanceBackend generation error for model '{self.model_name}': {exc}"
            ) from exc

        words = full_text.split(" ")
        for i, word in enumerate(words):
            token = word if i == len(words) - 1 else word + " "
            yield StreamEvent(
                type="output",
                token=token,
                backend=self.name,
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        yield StreamEvent(
            type="complete",
            content=full_text,
            backend=self.name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
