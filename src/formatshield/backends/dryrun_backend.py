"""
DryRunBackend — a deterministic, zero-dependency backend for CI testing and demos.

This backend implements the :class:`~formatshield.backends.protocol.Backend` protocol
without requiring any API keys, network access, or GPU.  It returns structurally
valid responses derived from the input schema, making it suitable for:

* CI pipeline runs where live API keys are unavailable
* Unit testing the TTF engine
* Demos and documentation examples
* Development iteration without incurring API costs

.. note::
   Responses are *structurally* valid but semantically meaningless.
   Use a live backend for production use.

Example::

    from formatshield.backends.dryrun_backend import DryRunBackend
    import formatshield as fs

    backend = DryRunBackend(seed=42)
    shield = fs.FormatShield(model="dryrun/test", backend=backend)
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator
from typing import Any

from formatshield.scorer.features import StreamEvent


class DryRunBackend:
    """
    Deterministic, zero-dependency backend for CI and testing.

    Implements the :class:`~formatshield.backends.protocol.Backend` protocol
    without any external API calls.  All responses are computed from the input
    schema and a seeded random number generator so test runs are reproducible.

    Parameters
    ----------
    seed:
        RNG seed for deterministic response generation.  Default ``42``.
    ttf_accuracy:
        Probability (0–1) that TTF responses contain a plausible ``final_answer``
        field when the schema requires one.  Default ``0.80``.
    direct_accuracy:
        Probability (0–1) for direct generation responses.  Default ``0.55``.
    base_latency_ms:
        Simulated base latency added to all responses, in milliseconds.
        Set to ``0`` to disable latency simulation.  Default ``5.0``.

    Example::

        backend = DryRunBackend(seed=0, ttf_accuracy=0.9, direct_accuracy=0.6)
        assert backend.supports_kv_cache_reuse is False
        result = await backend.generate("What is 2+2?")
        assert isinstance(result, str)
    """

    #: Backend identifier — shows up in benchmark result records.
    name: str = "dryrun"

    def __init__(
        self,
        seed: int = 42,
        ttf_accuracy: float = 0.80,
        direct_accuracy: float = 0.55,
        base_latency_ms: float = 5.0,
    ) -> None:
        self._rng = random.Random(seed)  # noqa: S311 — seeded RNG for determinism, not cryptography
        self._ttf_accuracy = ttf_accuracy
        self._direct_accuracy = direct_accuracy
        self._base_latency_ms = base_latency_ms
        self._call_count = 0

    # ------------------------------------------------------------------
    # Capability properties (Backend protocol)
    # ------------------------------------------------------------------

    @property
    def supports_kv_cache_reuse(self) -> bool:
        """DryRunBackend does not simulate KV-cache prefix reuse."""
        return False

    @property
    def accuracy_loss_baseline(self) -> float | None:
        """
        Simulated accuracy-loss baseline — the gap between ``direct_accuracy``
        and a perfect score of 1.0.
        """
        return round(1.0 - self._direct_accuracy, 4)

    @property
    def supports_logit_bias(self) -> bool:
        """DryRunBackend does not simulate logit bias."""
        return False

    # ------------------------------------------------------------------
    # Core generation methods (Backend protocol)
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
        """
        Return a deterministic structured response without making any API call.

        The response strategy depends on the *constraints* and *schema* arguments:

        * ``constraints is None and schema is None`` — thinking-style response with
          ``<think>…</think>`` tags (TTF Pass 1 pattern).
        * ``constraints == "json"`` or *schema* provided — JSON object generated
          from the schema structure.

        Parameters
        ----------
        prompt:
            The user prompt (used to seed per-call determinism).
        schema:
            Optional JSON schema dict.  Used to build a structurally valid
            response object.
        constraints:
            Pass ``"json"`` to signal that a JSON response is expected.
        kv_cache_prefix:
            Ignored; DryRunBackend does not simulate prefix caching.
        temperature:
            Ignored; DryRunBackend uses deterministic generation.
        top_p:
            Ignored; DryRunBackend uses deterministic generation.
        top_k:
            Ignored; DryRunBackend uses deterministic generation.
        max_tokens:
            Ignored; DryRunBackend uses deterministic generation.
        seed:
            Ignored; DryRunBackend uses its own internal RNG seed.
        frequency_penalty:
            Ignored; DryRunBackend uses deterministic generation.
        presence_penalty:
            Ignored; DryRunBackend uses deterministic generation.
        stop:
            Ignored; DryRunBackend uses deterministic generation.

        Returns
        -------
        str
            A deterministic response string.
        """
        self._call_count += 1
        # Yield to the event loop so async callers get cooperative behaviour.
        await asyncio.sleep(self._base_latency_ms / 1000.0)

        # TTF Pass 1: no constraints, no schema → return thinking text.
        if constraints is None and schema is None:
            return self._thinking_response(prompt)

        # JSON output path: build a response from the schema.
        return self._json_response(prompt, schema)

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
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream the response as an async iterator of
        :class:`~formatshield.scorer.features.StreamEvent`.

        Yields one ``"output"`` event per character of the response (up to 40
        characters) followed by a final ``"complete"`` event.

        Parameters
        ----------
        prompt:
            The user prompt.
        schema:
            Optional JSON schema dict.
        constraints:
            Pass ``"json"`` to signal a JSON response is expected.

        Yields
        ------
        StreamEvent
            Incremental ``output`` events followed by a single ``complete``.
        """
        return self._stream_impl(prompt, schema, constraints)

    async def _stream_impl(
        self,
        prompt: str,
        schema: dict | None,  # type: ignore[type-arg]
        constraints: str | None,
    ) -> AsyncIterator[StreamEvent]:
        response = await self.generate(prompt, schema=schema, constraints=constraints)

        # Stream individual characters (cap at 40 to keep tests fast).
        for i, char in enumerate(response[:40]):
            yield StreamEvent(
                type="output",
                token=char,
                backend=self.name,
                latency_ms=float(i + 1),
            )

        # Final complete event
        parsed_json: dict | None = None  # type: ignore[type-arg]
        if constraints == "json" or schema is not None:
            try:
                parsed_json = json.loads(response)
            except (json.JSONDecodeError, ValueError):
                pass

        yield StreamEvent(
            type="complete",
            content=response,
            json=parsed_json,
            backend=self.name,
            latency_ms=float(min(40, len(response)) + 10),
        )

    # ------------------------------------------------------------------
    # Response construction helpers
    # ------------------------------------------------------------------

    def _thinking_response(self, prompt: str) -> str:
        """Return a thinking-style response with ``<think>`` tags.

        # Pattern inspired by: CRANE research TTF implementation
        """
        steps = [
            "Identify the key quantities in the problem.",
            "Determine the relationships between them.",
            "Apply the relevant formula or algorithm.",
            "Verify the result by substitution.",
        ]
        thinking_text = " ".join(steps)
        return f"<think>{thinking_text}</think>\nI have reasoned through the problem carefully."

    def _json_response(
        self,
        prompt: str,
        schema: dict | None,  # type: ignore[type-arg]
    ) -> str:
        """Return a structurally valid JSON string.

        Uses the schema's ``properties`` to infer field names and types.
        Falls back to a generic ``{"result": "dryrun"}`` object when no schema
        is provided.
        """
        if schema is None:
            return json.dumps({"result": "dryrun_response", "confidence": 0.95})

        try:
            obj = self._generate_from_schema(schema)
        except Exception:
            obj = {"result": "dryrun_response"}

        return json.dumps(obj)

    def _generate_from_schema(self, schema: dict) -> Any:  # type: ignore[type-arg]
        """Recursively generate a value that matches *schema* structurally."""
        schema_type = schema.get("type", "object")

        if schema_type == "object":
            props = schema.get("properties") or {}
            return {key: self._generate_from_schema(val) for key, val in props.items()}

        if schema_type == "array":
            items_schema = schema.get("items") or {"type": "string"}
            # Return a single-element array to satisfy ``minItems`` constraints.
            return [self._generate_from_schema(items_schema)]

        if schema_type == "string":
            # Respect enum if present.
            if schema.get("enum"):
                return schema["enum"][0]
            return "dryrun"

        if schema_type in ("number", "float"):
            minimum = schema.get("minimum", 0.0)
            return float(minimum)

        if schema_type == "integer":
            minimum = schema.get("minimum", 0)
            return int(minimum)

        if schema_type == "boolean":
            return True

        if schema_type == "null":
            return None

        # anyOf / oneOf fallback
        for key in ("anyOf", "oneOf", "allOf"):
            candidates = schema.get(key)
            if candidates:
                return self._generate_from_schema(candidates[0])

        return "dryrun_value"

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        """Total number of :meth:`generate` calls made to this instance."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset :attr:`call_count` to zero."""
        self._call_count = 0
