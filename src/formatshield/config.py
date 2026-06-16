"""FormatShield extraction configuration.

Exposes ``ExtractionConfig`` — the per-call settings that control chunking
ratios, retry rounds, model selection, and more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ExtractionConfig"]

# ---------------------------------------------------------------------------
# Named constants — never use magic numbers inline
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT_UTILIZATION_RATIO: float = 0.50
DEFAULT_MAX_RETRY_ROUNDS: int = 2
# Per-call retry budget for transient failures (429/5xx/timeout); distinct from
# max_retry_rounds (field re-extraction). A TPM limit is a rolling 60s window, so a
# leaf must outlast it: 60s / ~6s mean backoff ⇒ 10 attempts, each honoring Retry-After.
DEFAULT_MAX_API_RETRIES: int = 10
# Max leaf extraction calls in flight at once. Firing every leaf of a round
# simultaneously overwhelms provider rate limits (429 storms) and, with retries,
# spirals into far more calls. A bounded concurrency window smooths the burst —
# the standard token-bucket / max-concurrency control for batched LLM calls
# (rate-limiting systems, arXiv:2602.11741; LLM batching guidance). Conservative
# default for free tiers; raise it on higher-throughput plans.
DEFAULT_MAX_CONCURRENT_CALLS: int = 4
# Recovery re-decomposes finer than the primary pass: fields that failed the first
# attempt are re-packed at this fraction of the reliability budget, so the retry
# leaves are smaller and more reliable. Closed-loop "smallest subtask + error
# correction" (MAKER, arXiv:2511.09030; adaptive granularity, arXiv:2510.17922).
DEFAULT_RECOVERY_BUDGET_SHRINK: float = 0.5
# Floor on the shrunk recovery budget (in difficulty-weighted units), so finer
# decomposition never collapses to absurdly tiny single-field calls.
MIN_RECOVERY_FIELDS_PER_CALL: int = 10
# Hard cap on fields per leaf (one API call), enforced as a difficulty-weighted load
# (see _reliability_load): ~50 easy fields, fewer hard ones. Cramming many fields into
# one call degrades reliability — instruction-following accuracy falls as the number
# of simultaneous instructions rises (IFScale, arXiv:2507.11538: even frontier models
# drop to ~68% at 500 instructions, biased toward earlier ones), and relevant content
# in a long packed context is missed (Lost-in-the-Middle, arXiv:2307.03172). The value
# 50 itself is a production heuristic, not a measured constant from a specific paper;
# the dynamic per-model calibration that would replace it is deferred.
# Yields K = O(load / cap) small, reliably-extractable leaves.
DEFAULT_MAX_FIELDS_PER_CALL: int = 50
DEFAULT_Z_TARGET: float = 1.645
DEFAULT_THINK_PHASE_BUDGET_MIN: int = 100
DEFAULT_THINK_PHASE_BUDGET_MAX: int = 150
DEFAULT_EVIDENCE_SCORE_THRESHOLD: float = 0.3


# ---------------------------------------------------------------------------
# ExtractionConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """Per-call configuration for the FormatShield extraction pipeline.

    All parameters are keyword-only and have sensible defaults so a
    ``ExtractionConfig()`` with no arguments is fully functional.

    Args:
        default_model: LLM model string to use when no model is specified
            per-call, e.g. ``"groq/llama-3.3-70b-versatile"``. ``None``
            means the caller must supply a model at call time.
        context_utilization_ratio: Fraction of the model's context window
            to use for document chunks. Range (0, 1]. Default 0.50.
        max_retry_rounds: Maximum number of extraction retry rounds for
            low-confidence or missing fields. Default 2.
        z_target: Z-score target used by the N-field routing engine to
            compute the minimum chunk count K_min. Default 1.645 (95th
            percentile).
        confidence_thresholds: Mapping of tier label → minimum confidence
            score, e.g. ``{"HIGH": 0.9, "MEDIUM": 0.7}``.
        document_language: BCP-47 language tag of the input document.
            Default ``"en"``.
        think_phase_budget: ``(min, max)`` token budget for the thinking
            phase. Default ``(100, 150)``.
        evidence_score_threshold: Minimum lexical/semantic evidence score
            for a chunk to be included in extraction context. Default 0.3.
        use_advanced_sfr: Enable advanced Semantic Field Routing (SFR)
            for improved precision on large schemas. Default ``False``.
        inject_dependencies: When ``True`` (the default), a dependent leaf's
            prompt receives a ``[Resolved dependency values]`` block with the
            values its upstream dependency fields produced in earlier rounds, and
            capacity packing reserves the tokens for that block. It is a no-op for
            schemas without cross-leaf dependencies. Set ``False`` to restore
            ordering-only dependency handling. Default ``True``.
        cascade_dependency_invalidation: When ``True`` **and**
            ``inject_dependencies`` is also ``True``, if a retry round changes a
            value that other fields depend on, those dependents are flagged
            ``NEEDS_REVALIDATION`` (CADTR). It is a no-op without
            ``inject_dependencies`` — a dependent is only stale if it consumed
            the upstream value via injection. Default ``False``.
        knowledge_fallback: When ``True``, fields the document does not state may
            be filled from the model's own knowledge instead of left ``NULL``. Best
            for well-known subject matter; risks unsourced values on private
            documents. Default ``False``.
        max_fields_per_call: Per-leaf reliability budget in difficulty-weighted
            units. A leaf grows while ``Σ (1 + λ·D(f)) <= max_fields_per_call``,
            where ``D(f)`` is each field's difficulty: a trivial field costs ~1
            unit, a hard one more. So a leaf holds up to ~``max_fields_per_call``
            easy fields, or fewer hard ones — bounding by reliability, not raw
            count or token budget alone (a large window cannot cram hundreds of
            fields into one unreliable call). Default 50 (the production
            reliability heuristic); forces K = O(load / budget) small leaves.
        recovery_budget_shrink: Fraction of the reliability budget used when the
            recovery pass re-packs fields that failed the first attempt. < 1 makes
            recovery decompose FINER (smaller, more reliable leaves) where the
            primary pass struggled — closed-loop "smallest subtask + error
            correction". Default 0.5; floored at ``MIN_RECOVERY_FIELDS_PER_CALL``.
        max_concurrent_calls: Maximum leaf extraction calls in flight at once.
            Bounds the concurrency of each execution round so a wide schema does
            not fire dozens of calls simultaneously and trip provider rate limits.
            Default 4 (safe for free tiers); raise for higher-throughput plans.
        max_api_retries: Per-call retry budget for transient failures (429 / 5xx /
            timeout), honoring Retry-After. Distinct from ``max_retry_rounds``
            (field re-extraction). Default 10 (outlasts a rolling-window TPM storm);
            lower for fail-fast. Must be > 0.

    Example:
        >>> cfg = ExtractionConfig(default_model="groq/llama-3.1-8b")
        >>> cfg.context_utilization_ratio
        0.5
        >>> cfg.max_retry_rounds
        2
    """

    default_model: str | None = None
    context_utilization_ratio: float = DEFAULT_CONTEXT_UTILIZATION_RATIO
    max_retry_rounds: int = DEFAULT_MAX_RETRY_ROUNDS
    z_target: float = DEFAULT_Z_TARGET
    confidence_thresholds: dict[str, float] = field(
        default_factory=lambda: {"HIGH": 0.9, "MEDIUM": 0.7}
    )
    document_language: str = "en"
    think_phase_budget: tuple[int, int] = (
        DEFAULT_THINK_PHASE_BUDGET_MIN,
        DEFAULT_THINK_PHASE_BUDGET_MAX,
    )
    evidence_score_threshold: float = DEFAULT_EVIDENCE_SCORE_THRESHOLD
    use_advanced_sfr: bool = False
    inject_dependencies: bool = True
    cascade_dependency_invalidation: bool = False
    knowledge_fallback: bool = False
    max_fields_per_call: int = DEFAULT_MAX_FIELDS_PER_CALL
    recovery_budget_shrink: float = DEFAULT_RECOVERY_BUDGET_SHRINK
    max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS
    max_api_retries: int = DEFAULT_MAX_API_RETRIES
    # When True, validate values exactly as extracted (no lenient normalization of
    # formatted numbers/booleans). Default lenient: accept "$1,234,568" as 1234568.
    strict_validation: bool = False
