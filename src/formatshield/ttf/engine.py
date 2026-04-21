"""TTF (Think-Then-Format) Engine — two-pass generation following CRANE (arXiv 2502.09061).

The engine implements the core insight from CRANE: unconstrained reasoning (Pass 1)
followed by constrained formatting (Pass 2) recovers accuracy lost to FSM-based
token masking in direct structured generation.

KV cache strategy:
    vLLM  — native prefix caching: Pass 2 reuses Pass 1 KV activations → <10% overhead
    others — simulated: Pass 1 output prepended as context prefix for Pass 2 → ~20–40%
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from formatshield._retry import FailedAttempt, FormatShieldRetryException, build_reask_prompt
from formatshield.scorer.features import StreamEvent
from formatshield.ttf.prompts import (
    _collect_schema_field_info,
    build_cache_prefix_for_format_prompt,
    build_format_prompt,
    build_schema_phi_think_prompt,
    build_think_prompt,
    extract_thinking,
)
from formatshield.ttf.quality_gate import score_thinking_trace
from formatshield.ttf.low_latency_ttf import (
    LowLatencyConfig,
    compute_low_latency_thinking_budget,
    get_selective_reasoning_focus,
)

if TYPE_CHECKING:
    from formatshield.backends.protocol import Backend
    from formatshield.oracle.routing_score import RoutingScore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Φ-Proportional Thinking Budget
# ---------------------------------------------------------------------------


def _phi_thinking_budget(phi: float) -> int:
    """Map routing score Φ to a Pass 1 max-token budget.

    Simple schemas (low Φ) get a small budget to avoid wasted tokens.
    Complex, highly-coupled schemas (high Φ) get a generous budget so the
    model has room for deep multi-step reasoning.

    Thresholds:
    - Φ ≥ 0.90 → 4096 tokens  (MAXIMUM depth)
    - Φ ∈ [0.75, 0.90) → 1024 tokens  (DEEP / STANDARD)
    - Φ ∈ [0.65, 0.75) → 512 tokens   (LIGHT)
    - Φ < 0.65 → 256 tokens            (minimal; TTF rarely fires below 0.65)
    """
    if phi >= 0.90:
        return 4096
    if phi >= 0.75:
        return 1024
    if phi >= 0.65:
        return 512
    return 256


# ---------------------------------------------------------------------------
# Self-consistency helpers
# ---------------------------------------------------------------------------

#: Φ threshold above which self-consistency Pass 1 is triggered automatically.
_SC_PHI_THRESHOLD: float = 0.95

#: Default number of parallel Pass 1 traces for self-consistency mode.
DEFAULT_SC_K: int = 3


async def _run_self_consistency_pass1(
    backend: Backend,
    think_prompt: str,
    k: int,
    max_tokens: int | None,
    logit_bias: dict[int, float] | None,
    schema: dict[str, Any] | None,
    routing_score: RoutingScore | None,
) -> tuple[str, str]:
    """Run *k* parallel Pass 1 traces and return the best ``(thinking_text, raw_thinking)``.

    Each trace is scored by the Pass 1 quality gate.  The trace with the
    highest score is returned.  When ``k == 1`` the function degenerates to
    a single call (no overhead).

    Parameters
    ----------
    backend:
        Inference backend to call.
    think_prompt:
        The fully-built Pass 1 prompt (schema-aware or generic).
    k:
        Number of parallel traces to generate.  Must be ≥ 1.
    max_tokens:
        Token budget per trace (from Φ-proportional budget).
    logit_bias:
        Optional logit-bias map for field-name nudging.
    schema:
        JSON schema used for quality-gate scoring.  ``None`` skips gate scoring.
    routing_score:
        Routing score used by the quality gate.  ``None`` skips gate scoring.

    Returns
    -------
    tuple[str, str]
        ``(thinking_text, raw_thinking)`` of the best-scoring trace.
    """
    if k <= 1:
        raw = await backend.generate(
            think_prompt,
            schema=None,
            constraints=None,
            max_tokens=max_tokens,
            logit_bias=logit_bias,
        )
        return extract_thinking(raw), raw

    # Launch k parallel Pass 1 calls
    tasks = [
        backend.generate(
            think_prompt,
            schema=None,
            constraints=None,
            max_tokens=max_tokens,
            logit_bias=logit_bias,
        )
        for _ in range(k)
    ]
    results: list[str] = await asyncio.gather(*tasks, return_exceptions=False)

    best_raw = results[0]
    best_thinking = extract_thinking(best_raw)
    best_score = 0.0

    for raw in results:
        thinking = extract_thinking(raw)
        if schema is not None and routing_score is not None:
            gate = score_thinking_trace(thinking, schema, routing_score)
            score = gate.score
        else:
            # No schema — use trace length as a tiebreaker (longer = more reasoning)
            score = float(len(thinking))

        if score > best_score:
            best_score = score
            best_raw = raw
            best_thinking = thinking

    logger.debug(
        "TTFEngine: self-consistency selected best trace (k=%d best_score=%.3f)",
        k,
        best_score,
    )
    return best_thinking, best_raw


# ---------------------------------------------------------------------------
# Schema-vocabulary logit biasing
# ---------------------------------------------------------------------------


def _build_schema_logit_bias(
    schema_fields: list[str],
    encoding_name: str = "cl100k_base",
    bias_value: float = 2.0,
) -> dict[int, float]:
    """Build a logit-bias mapping that nudges the model toward schema field names.

    Each schema field name is tokenised and the first token of each name is
    given a positive bias so Pass 1 reasoning is more likely to mention all
    required field names — improving required-field coverage in the quality gate.

    Parameters
    ----------
    schema_fields:
        List of field name strings extracted from the target schema.
    encoding_name:
        tiktoken encoding to use for tokenisation.  Defaults to ``cl100k_base``
        which covers GPT-4 / LLaMA-3 / Mistral token vocabularies.
    bias_value:
        Logit bias value (additive, in log-probability space).  ``2.0`` is a
        gentle nudge; values above ``5.0`` can distort output quality.

    Returns
    -------
    dict[int, float]
        Mapping of ``{token_id: bias_value}`` suitable for the backend's
        ``logit_bias`` parameter.  Empty dict if tiktoken is not installed.
    """
    try:
        import tiktoken  # optional dependency

        enc = tiktoken.get_encoding(encoding_name)
    except ImportError:
        logger.debug("_build_schema_logit_bias: tiktoken not installed — skipping")
        return {}
    except Exception as exc:
        logger.debug("_build_schema_logit_bias: encoding error — %s", exc)
        return {}

    bias: dict[int, float] = {}
    for field_name in schema_fields:
        if not field_name:
            continue
        tokens = enc.encode(field_name)
        if tokens:
            # Bias only the first token of the field name to avoid bloating the map
            bias[tokens[0]] = bias_value
    return bias


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

    # Maximum reask attempts before giving up: 2 means up to 3 total generations
    # (1 original + 2 reasks) to avoid runaway token costs.
    DEFAULT_MAX_REASKS: int = 2

    def __init__(
        self,
        backend: Backend,
        ttf_fallback: bool = True,
        expose_thinking: bool = False,
        max_reasks: int = DEFAULT_MAX_REASKS,
        ttf_self_consistency: int = 1,
        low_latency_mode: bool = False,
        low_latency_config: LowLatencyConfig | None = None,
    ) -> None:
        self._backend = backend
        self._ttf_fallback = ttf_fallback
        self._expose_thinking = expose_thinking
        self._max_reasks = max_reasks
        # K for self-consistency: 1 = disabled, ≥2 = run K parallel Pass 1 traces
        self._ttf_self_consistency: int = max(1, ttf_self_consistency)
        self._low_latency_mode = low_latency_mode
        self._low_latency_config = low_latency_config or LowLatencyConfig()

    # ------------------------------------------------------------------
    # Primary generation method
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        schema_model: type[BaseModel] | None = None,
        kv_cache_prefix: str | None = None,
        routing_score: RoutingScore | None = None,
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
        routing_score:
            Optional :class:`~formatshield.oracle.routing_score.RoutingScore`
            from OracleX.  When provided together with *schema*, enables:

            * Schema-aware Pass 1 prompt that injects field dependency order,
              enum constraints, Φ routing depth signal, and vocabulary bridge
              hints for schema fields absent from the prompt vocabulary (ΔK gap).
            * τ-conditioned Pass 2 temperature that scales down for
              tight-constraint schemas (high τ) to reduce retries.

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
        # Use schema-aware prompt when routing signal is available
        if routing_score is not None and schema is not None:
            think_prompt = build_schema_phi_think_prompt(
                original_prompt=prompt,
                schema=schema,
                phi=routing_score.phi,
                tau=routing_score.tau,
                delta_k=routing_score.delta_k,
                lambda2=routing_score.lambda2,
            )
            logger.debug(
                "TTFEngine: Pass 1 — schema-aware prompt (Φ=%.3f τ=%.3f ΔK=%.3f backend=%s)",
                routing_score.phi,
                routing_score.tau,
                routing_score.delta_k,
                self._backend.name,
            )
        else:
            think_prompt = build_think_prompt(prompt)
            logger.debug(
                "TTFEngine: Pass 1 — unconstrained reasoning (backend=%s)", self._backend.name
            )

        # τ-conditioned Pass 2 temperature.
        # High τ (tight constraints: enums, booleans) → low temperature for precision.
        # Low τ (free text fields) → higher temperature for semantic richness.
        # Formula: temperature = max(0.05, 0.7 * (1 − τ))
        pass2_temperature: float | None = None
        if routing_score is not None:
            pass2_temperature = max(0.05, 0.7 * (1.0 - routing_score.tau))
            logger.debug(
                "TTFEngine: Pass 2 temperature τ-conditioned to %.3f (τ=%.3f)",
                pass2_temperature,
                routing_score.tau,
            )

        # Φ-proportional thinking budget.
        # Budget scales with routing Φ: low Φ → 256 tokens, high Φ → 4096.
        pass1_max_tokens: int | None = None
        if routing_score is not None:
            if self._low_latency_mode:
                pass1_max_tokens = compute_low_latency_thinking_budget(
                    routing_score.phi,
                    config=self._low_latency_config,
                )
            else:
                pass1_max_tokens = _phi_thinking_budget(routing_score.phi)
            logger.debug(
                "TTFEngine: Pass 1 budget — %d tokens (Φ=%.3f)",
                pass1_max_tokens,
                routing_score.phi,
            )

        if self._low_latency_mode and routing_score is not None and schema is not None:
            selective_focus = get_selective_reasoning_focus(schema, routing_score.tau)
            if selective_focus:
                think_prompt = f"{think_prompt}\n\n{selective_focus}"
                logger.debug("TTFEngine: low-latency selective reasoning focus enabled")

        # ------------------------------------------------------------------
        # Pass 1: unconstrained reasoning
        # ------------------------------------------------------------------

        # Build soft logit bias toward schema field-name tokens when the backend
        # supports it.  This nudges Pass 1 reasoning to mention all required fields,
        # improving quality-gate field-coverage scores.
        pass1_logit_bias: dict[int, float] | None = None
        if schema is not None and getattr(self._backend, "supports_logit_bias", False):
            field_infos = _collect_schema_field_info(schema)
            field_names = [f["path"].split(".")[-1] for f in field_infos]
            bias_map = _build_schema_logit_bias(field_names)
            if bias_map:
                pass1_logit_bias = bias_map
                logger.debug(
                    "TTFEngine: Pass 1 — logit bias active for %d field tokens",
                    len(bias_map),
                )

        # Determine self-consistency K: explicit setting OR auto-trigger when Φ ≥ 0.95.
        sc_k = self._ttf_self_consistency
        if sc_k < 2 and routing_score is not None and routing_score.phi >= _SC_PHI_THRESHOLD:
            sc_k = DEFAULT_SC_K
            logger.debug(
                "TTFEngine: auto-enabling self-consistency K=%d (Φ=%.3f ≥ %.2f)",
                sc_k,
                routing_score.phi,
                _SC_PHI_THRESHOLD,
            )

        thinking_text, raw_thinking = await _run_self_consistency_pass1(
            backend=self._backend,
            think_prompt=think_prompt,
            k=sc_k,
            max_tokens=pass1_max_tokens,
            logit_bias=pass1_logit_bias,
            schema=schema,
            routing_score=routing_score,
        )
        logger.debug(
            "TTFEngine: Pass 1 complete — %d chars of thinking (k=%d)", len(thinking_text), sc_k
        )

        # Pass 1 quality gate.
        # Score the thinking trace; retry once if it fails the heuristic checks.
        if routing_score is not None and schema is not None:
            gate_result = score_thinking_trace(thinking_text, schema, routing_score)
            if not gate_result.passed:
                logger.warning(
                    "TTFEngine: Pass 1 quality gate FAILED (score=%.2f, failed=%s) — retrying",
                    gate_result.score,
                    gate_result.failed_checks,
                )
                thinking_text, raw_thinking = await _run_self_consistency_pass1(
                    backend=self._backend,
                    think_prompt=think_prompt,
                    k=sc_k,
                    max_tokens=pass1_max_tokens,
                    logit_bias=pass1_logit_bias,
                    schema=schema,
                    routing_score=routing_score,
                )
                gate_result_retry = score_thinking_trace(thinking_text, schema, routing_score)
                if not gate_result_retry.passed:
                    logger.warning(
                        "TTFEngine: Pass 1 quality gate FAILED after retry "
                        "(score=%.2f) — continuing with degraded trace",
                        gate_result_retry.score,
                    )
                else:
                    logger.debug(
                        "TTFEngine: Pass 1 quality gate passed after retry (score=%.2f)",
                        gate_result_retry.score,
                    )

        # ------------------------------------------------------------------
        # Pass 2: constrained formatting
        # ------------------------------------------------------------------
        format_prompt = build_format_prompt(think_prompt, raw_thinking, schema=schema)

        # Build a static, schema-derived cache prefix for Pass 2 so backends
        # with KV-cache prefix reuse can amortise the encoding cost of the
        # format instruction across many requests that share the same schema.
        pass2_kv_prefix: str | None = None
        if self._backend.supports_kv_cache_reuse:
            if kv_cache_prefix is not None:
                pass2_kv_prefix = kv_cache_prefix
            elif schema is not None:
                pass2_kv_prefix = build_cache_prefix_for_format_prompt(schema)
            else:
                pass2_kv_prefix = think_prompt
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
            temperature=pass2_temperature,
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

    def stream(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Return an async iterator of StreamEvent for two-pass TTF generation.

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
        """Validate *json_output* against *schema_model*, with reask on failure.

        On validation failure:

        1. If ``self._max_reasks > 0``: constructs a reask prompt that feeds
           the failed output + error back to the model and retries up to
           ``self._max_reasks`` times.  Each failed attempt is recorded as a
           :class:`~formatshield._retry.FailedAttempt`.
        2. If all reasks fail and ``self._ttf_fallback`` is ``True``: falls
           back to :meth:`generate_direct` (single-pass constrained generation).
        3. If ``self._ttf_fallback`` is ``False``: returns the last invalid
           output as-is.

        Returns
        -------
        tuple[str, bool]
            ``(output_text, fallback_triggered)`` where ``fallback_triggered``
            is ``True`` only when the single-pass fallback path was used.

        Raises
        ------
        FormatShieldRetryException
            When all reasks fail AND ``ttf_fallback=False`` and you want to
            surface the full attempt history to the caller.  Currently this
            exception is only raised internally; callers receive the last raw
            output instead.
        """
        failed_attempts: list[FailedAttempt] = []
        current_output = json_output
        current_prompt = prompt

        for attempt_number in range(1, self._max_reasks + 2):
            # Attempt validation
            try:
                schema_model.model_validate_json(current_output)
                return current_output, False  # ← success
            except (ValidationError, json.JSONDecodeError) as exc:
                logger.warning(
                    "TTFEngine: attempt %d/%d — Pass 2 output failed validation — %s",
                    attempt_number,
                    self._max_reasks + 1,
                    exc,
                )
                failed_attempts.append(
                    FailedAttempt(
                        attempt_number=attempt_number,
                        exception=exc,
                        raw_output=current_output,
                        reask_prompt=current_prompt,
                    )
                )

            # Can we reask?
            reasks_used = attempt_number - 1
            if reasks_used < self._max_reasks:
                # Build a reask prompt: original + failed output + error
                current_prompt = build_reask_prompt(
                    original_prompt=prompt,
                    failed_output=current_output,
                    error=failed_attempts[-1].exception,
                    schema=schema,
                )
                logger.info(
                    "TTFEngine: reask %d/%d — sending corrective prompt",
                    reasks_used + 1,
                    self._max_reasks,
                )
                try:
                    current_output = await self._backend.generate(
                        current_prompt,
                        schema=schema,
                        constraints="json",
                    )
                except Exception as exc:
                    logger.error("TTFEngine: reask backend call failed — %s", exc)
                    break  # give up on reasks, try direct fallback

        # All reasks exhausted — record as FormatShieldRetryException internally
        retry_exc = FormatShieldRetryException(
            f"All {len(failed_attempts)} attempt(s) failed schema validation",
            failed_attempts=failed_attempts,
        )
        logger.warning("TTFEngine: %s", retry_exc)

        if not self._ttf_fallback:
            logger.warning("TTFEngine: ttf_fallback=False — returning last invalid output")
            return current_output, False

        # Final fallback: single-pass direct generation
        logger.info("TTFEngine: falling back to direct generation after reask exhaustion")
        try:
            direct_output = await self.generate_direct(prompt, schema=schema)
            try:
                schema_model.model_validate_json(direct_output)
            except (ValidationError, json.JSONDecodeError) as exc:
                logger.warning("TTFEngine: fallback direct output also failed validation — %s", exc)
            return direct_output, True
        except Exception as exc:
            logger.error("TTFEngine: fallback direct generation failed — %s", exc)
            return current_output, True
