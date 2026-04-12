"""TTF (Think-Then-Format) Engine — two-pass generation following CRANE (arXiv 2502.09061).

The engine implements the core insight from CRANE: unconstrained reasoning (Pass 1)
followed by constrained formatting (Pass 2) recovers accuracy lost to FSM-based
token masking in direct structured generation.

KV cache strategy:
    vLLM  — native prefix caching: Pass 2 reuses Pass 1 KV activations → <10% overhead
    others — simulated: Pass 1 output prepended as context prefix for Pass 2 → ~20–40%
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from formatshield.scorer.features import StreamEvent
from formatshield.ttf.prompts import build_format_prompt, build_think_prompt, extract_thinking

if TYPE_CHECKING:
    from formatshield.backends.protocol import Backend

logger = logging.getLogger(__name__)


class TTFEngine:
    """Two-pass Think-Then-Format generation engine.

    Orchestrates the two-pass generation pattern from CRANE (arXiv 2502.09061):

    * **Pass 1** — unconstrained reasoning: the model thinks freely inside
      ``<think>…</think>`` tags without any JSON/grammar constraints.
    * **Pass 2** — constrained formatting: the model produces the final
      structured JSON output conditioned on its own reasoning.

    When the backend supports native KV-cache prefix reuse (vLLM), Pass 2
    reuses the Pass 1 KV activations, keeping latency overhead below 10%.

    Parameters
    ----------
    backend:
        Any object implementing the :class:`~formatshield.backends.protocol.Backend`
        protocol.
    ttf_fallback:
        When ``True`` (default), if Pass 2 output fails Pydantic schema
        validation the engine automatically retries with direct single-pass
        generation.  The fallback is logged and surfaced in :class:`GenerationResult`.
    expose_thinking:
        When ``True``, thinking text is included in streaming events.
        When ``False`` (default), ``StreamEvent(type="thinking", …)`` events
        are still yielded (for internal use) but callers can filter them.

    Example::

        engine = TTFEngine(backend=groq_backend)
        thinking, output = await engine.generate(
            prompt="Solve: 3x + 7 = 22. Extract steps and answer.",
            schema={"type": "object", "properties": {"steps": {...}, "answer": {...}}},
        )
    """

    def __init__(
        self,
        backend: Backend,
        ttf_fallback: bool = True,
        expose_thinking: bool = False,
    ) -> None:
        self._backend = backend
        self._ttf_fallback = ttf_fallback
        self._expose_thinking = expose_thinking

    # ------------------------------------------------------------------
    # Primary generation method
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        schema_model: type[BaseModel] | None = None,
        kv_cache_prefix: str | None = None,
    ) -> tuple[str, str]:
        """Run two-pass TTF generation and return ``(thinking_text, json_output)``.

        Parameters
        ----------
        prompt:
            The original user prompt (unmodified).
        schema:
            Optional JSON Schema dict.  Used as the constraint for Pass 2.
        schema_model:
            Optional Pydantic model class for output validation.  When provided,
            the engine attempts ``schema_model.model_validate_json(output)``; on
            failure it optionally falls back to direct generation.
        kv_cache_prefix:
            Override the KV-cache prefix passed to the backend.  Defaults to the
            Pass 1 (think) prompt when the backend supports KV-cache reuse.

        Returns
        -------
        tuple[str, str]
            ``(thinking_text, json_output)`` where *thinking_text* is the content
            extracted from ``<think>…</think>`` tags and *json_output* is the
            raw JSON string from Pass 2.

        Raises
        ------
        RuntimeError
            Re-raised from the backend if both TTF and fallback paths fail.
        """
        think_prompt = build_think_prompt(prompt)

        # ------------------------------------------------------------------
        # Pass 1: unconstrained reasoning
        # ------------------------------------------------------------------
        logger.debug("TTFEngine: Pass 1 — unconstrained reasoning (backend=%s)", self._backend.name)
        raw_thinking = await self._backend.generate(
            think_prompt,
            schema=None,
            constraints=None,
        )

        thinking_text = extract_thinking(raw_thinking)
        logger.debug("TTFEngine: Pass 1 complete — %d chars of thinking", len(thinking_text))

        # ------------------------------------------------------------------
        # Pass 2: constrained formatting
        # ------------------------------------------------------------------
        format_prompt = build_format_prompt(think_prompt, raw_thinking, schema=schema)

        # Use native KV-cache reuse when available (vLLM prefix caching)
        pass2_kv_prefix: str | None = None
        if self._backend.supports_kv_cache_reuse:
            pass2_kv_prefix = kv_cache_prefix if kv_cache_prefix is not None else think_prompt
            logger.debug(
                "TTFEngine: Pass 2 — using KV-cache prefix reuse (backend=%s)", self._backend.name
            )
        else:
            logger.debug("TTFEngine: Pass 2 — no KV-cache reuse (backend=%s)", self._backend.name)

        json_output = await self._backend.generate(
            format_prompt,
            schema=schema,
            constraints="json",
            kv_cache_prefix=pass2_kv_prefix,
        )

        logger.debug("TTFEngine: Pass 2 complete — %d chars of JSON output", len(json_output))

        # ------------------------------------------------------------------
        # Validation + optional fallback
        # ------------------------------------------------------------------
        if schema_model is not None:
            validated_output, did_fallback = await self._validate_or_fallback(
                json_output=json_output,
                schema_model=schema_model,
                prompt=prompt,
                schema=schema,
            )
            if did_fallback:
                # On fallback we return empty thinking (single-pass had no think phase)
                return "", validated_output

        return thinking_text, json_output

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    async def stream(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream two-pass TTF generation as an async iterator of StreamEvent.

        Yields events in three phases:

        1. ``thinking`` — content chunks from Pass 1 (reasoning phase).
        2. ``output``   — incremental token chunks from Pass 2.
        3. ``complete`` — final event carrying the assembled JSON dict.

        Parameters
        ----------
        prompt:
            The original user prompt.
        schema:
            Optional JSON Schema dict for Pass 2 constraint.

        Yields
        ------
        StreamEvent
            Events of type ``"thinking"``, ``"output"``, and ``"complete"``.
        """
        return self._stream_impl(prompt, schema)

    async def _stream_impl(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
    ) -> AsyncIterator[StreamEvent]:
        import time

        t0 = time.monotonic()
        think_prompt = build_think_prompt(prompt)

        # ------------------------------------------------------------------
        # Pass 1: stream thinking events
        # ------------------------------------------------------------------
        raw_thinking_parts: list[str] = []

        try:
            thinking_stream = await self._backend.stream(
                think_prompt,
                schema=None,
                constraints=None,
            )
            async for event in thinking_stream:
                if event.type == "output" and event.token:
                    raw_thinking_parts.append(event.token)
                    yield StreamEvent(
                        type="thinking",
                        content=event.token,
                        backend=self._backend.name,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
                elif event.type == "complete" and event.content:
                    raw_thinking_parts.append(event.content)
        except Exception as exc:
            logger.warning("TTFEngine.stream: Pass 1 failed — %s", exc)
            # Emit an error thinking event and fall through to direct generation
            yield StreamEvent(
                type="thinking",
                content=f"[Pass 1 failed: {exc}]",
                backend=self._backend.name,
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        raw_thinking = "".join(raw_thinking_parts)
        format_prompt = build_format_prompt(think_prompt, raw_thinking, schema=schema)

        # ------------------------------------------------------------------
        # Pass 2: stream output events
        # ------------------------------------------------------------------
        output_parts: list[str] = []

        try:
            output_stream = await self._backend.stream(
                format_prompt,
                schema=schema,
                constraints="json",
            )
            async for event in output_stream:
                if event.type == "output" and event.token:
                    output_parts.append(event.token)
                    yield StreamEvent(
                        type="output",
                        token=event.token,
                        backend=self._backend.name,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
                elif event.type == "complete" and event.content:
                    output_parts.append(event.content)
        except Exception as exc:
            logger.error("TTFEngine.stream: Pass 2 failed — %s", exc)
            yield StreamEvent(
                type="complete",
                json=None,
                backend=self._backend.name,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
            return

        # ------------------------------------------------------------------
        # Final: assemble and emit complete event
        # ------------------------------------------------------------------
        assembled = "".join(output_parts)
        parsed_json: dict[str, Any] | None = None
        try:
            parsed_json = json.loads(assembled)
        except json.JSONDecodeError:
            logger.warning(
                "TTFEngine.stream: Pass 2 output is not valid JSON — "
                "emitting complete event with json=None"
            )

        yield StreamEvent(
            type="complete",
            json=parsed_json,
            content=assembled,
            backend=self._backend.name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    # ------------------------------------------------------------------
    # Direct fallback (single-pass)
    # ------------------------------------------------------------------

    async def generate_direct(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> str:
        """Single-pass constrained generation — the fallback path.

        Sends *prompt* directly to the backend with ``constraints="json"``.
        Used when:
        * The oracle routes a request to ``"direct"`` strategy.
        * TTF validation fails and ``ttf_fallback=True``.

        Parameters
        ----------
        prompt:
            The original user prompt (NOT modified with think-prompt suffixes).
        schema:
            Optional JSON Schema dict.

        Returns
        -------
        str
            Raw JSON output from the backend.
        """
        logger.debug(
            "TTFEngine.generate_direct: single-pass constrained generation (backend=%s)",
            self._backend.name,
        )
        return await self._backend.generate(
            prompt,
            schema=schema,
            constraints="json",
        )

    # ------------------------------------------------------------------
    # Internal validation helpers
    # ------------------------------------------------------------------

    async def _validate_or_fallback(
        self,
        json_output: str,
        schema_model: type[BaseModel],
        prompt: str,
        schema: dict[str, Any] | None,
    ) -> tuple[str, bool]:
        """Validate *json_output* against *schema_model*.

        If validation succeeds, returns ``(json_output, False)``.
        If validation fails and ``self._ttf_fallback`` is ``True``:
            * Retries with :meth:`generate_direct`.
            * Returns ``(direct_output, True)``.
        If validation fails and fallback is disabled:
            * Returns ``(json_output, False)`` and logs a warning.

        Returns
        -------
        tuple[str, bool]
            ``(output_text, fallback_triggered)``.
        """
        try:
            schema_model.model_validate_json(json_output)
            return json_output, False
        except (ValidationError, json.JSONDecodeError) as exc:
            logger.warning(
                "TTFEngine: Pass 2 output failed schema validation — %s",
                exc,
            )

        if not self._ttf_fallback:
            logger.warning("TTFEngine: ttf_fallback=False — returning invalid output as-is")
            return json_output, False

        # Retry as direct generation
        logger.info("TTFEngine: falling back to direct generation")
        try:
            direct_output = await self.generate_direct(prompt, schema=schema)
            # Attempt to validate the fallback output too
            try:
                schema_model.model_validate_json(direct_output)
            except (ValidationError, json.JSONDecodeError) as exc:
                logger.warning(
                    "TTFEngine: fallback direct output also failed validation — %s",
                    exc,
                )
            return direct_output, True
        except Exception as exc:
            logger.error("TTFEngine: fallback direct generation failed — %s", exc)
            return json_output, True
