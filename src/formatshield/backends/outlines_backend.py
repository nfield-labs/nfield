"""Outlines constrained-decoding backend for FormatShield."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from formatshield.scorer.features import StreamEvent


class OutlinesBackend:
    """
    FormatShield backend that wraps the `outlines
    <https://outlines-dev.github.io/outlines/>`_ library for
    **constrained JSON schema generation**.

    Outlines intercepts the model's token-sampling loop and hard-masks any
    token that would violate the supplied JSON schema, making schema
    conformance a mathematical guarantee rather than a probabilistic hope.
    Because no valid token is ever suppressed and no invalid token ever
    accepted, accuracy loss relative to unconstrained generation is
    effectively zero.

    ``outlines`` is an *optional* dependency.  It is imported lazily inside
    each method so that the rest of FormatShield remains importable even when
    the library is not installed.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (e.g.
        ``"microsoft/Phi-3-mini-4k-instruct"``).
    device:
        PyTorch device string passed to the underlying transformers backend
        (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    #: Backend identifier consumed by the FormatShield router.
    name: str = "outlines"

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supports_kv_cache_reuse(self) -> bool:
        """
        Outlines does not expose server-side KV-cache prefix reuse.

        Constrained decoding is performed in-process via logit masking;
        there is no separate server that could cache key/value tensors
        across distinct requests.
        """
        return False

    @property
    def accuracy_loss_baseline(self) -> float:
        """
        Zero baseline accuracy loss.

        Outlines performs *exact* constrained decoding: the token sampler is
        hard-masked so that only schema-valid continuations are ever sampled.
        There is no stochastic approximation, so accuracy loss relative to an
        oracle that always produces valid JSON is definitionally 0.0.
        """
        return 0.0

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
        Generate a response using outlines constrained decoding and return
        the full text.

        When *schema* is provided the generation is constrained to the
        supplied JSON schema via ``outlines.generate.json``.  Without a
        schema, ``outlines.generate.text`` is used for unconstrained free
        text generation.

        Parameters
        ----------
        prompt:
            The user prompt passed directly to the generator.
        schema:
            Optional JSON schema dict.  When present, outlines constrains
            the sampler to produce only schema-valid JSON.
        constraints:
            Optional constraint hint string.  Not used by this backend
            beyond the *schema* check; reserved for router compatibility.
        kv_cache_prefix:
            Ignored; outlines does not support prefix caching.

        Returns
        -------
        str
            The model's response text.

        Raises
        ------
        ImportError
            When the ``outlines`` package is not installed.  Install it with
            ``pip install outlines``.
        RuntimeError
            Wraps any generation-time exception with a human-readable
            message.
        """
        try:
            import outlines  # type: ignore[import-untyped]
            import outlines.generate  # type: ignore[import-untyped]
            import outlines.models  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'outlines' package is required for OutlinesBackend. "
                "Install it with: pip install outlines"
            ) from exc

        try:
            model = outlines.models.transformers(self.model_name, device=self.device)
            if schema is not None:
                generator = outlines.generate.json(model, schema)
            else:
                generator = outlines.generate.text(model)
            result: str = generator(prompt)
        except Exception as exc:
            raise RuntimeError(
                f"OutlinesBackend generation error for model '{self.model_name}': {exc}"
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

        Outlines does not expose a native streaming API; this method
        simulates streaming by generating the complete text first and then
        yielding one ``"output"`` event per whitespace-delimited word,
        followed by a single ``"complete"`` event containing the full text.

        Parameters
        ----------
        prompt:
            The user prompt.
        schema:
            Optional JSON schema dict.  When present, outlines constrains
            the sampler to produce only schema-valid JSON.
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
            When the ``outlines`` package is not installed.  Install it with
            ``pip install outlines``.
        RuntimeError
            Wraps any generation-time exception with a human-readable
            message.
        """
        t0 = time.monotonic()

        try:
            import outlines  # type: ignore[import-untyped]
            import outlines.generate  # type: ignore[import-untyped]
            import outlines.models  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'outlines' package is required for OutlinesBackend. "
                "Install it with: pip install outlines"
            ) from exc

        try:
            model = outlines.models.transformers(self.model_name, device=self.device)
            if schema is not None:
                generator = outlines.generate.json(model, schema)
            else:
                generator = outlines.generate.text(model)
            full_text: str = generator(prompt)
        except Exception as exc:
            raise RuntimeError(
                f"OutlinesBackend generation error for model '{self.model_name}': {exc}"
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
