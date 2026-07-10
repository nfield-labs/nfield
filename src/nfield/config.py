"""nfield extraction configuration.

Exposes ``ExtractionConfig`` - the per-call settings that control chunking
ratios, retry rounds, model selection, and more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nfield.providers._cache import ResponseCache

__all__ = ["ExtractionConfig"]

# ---------------------------------------------------------------------------
# Named constants - never use magic numbers inline
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT_UTILIZATION_RATIO: float = 0.50
DEFAULT_MAX_RETRY_ROUNDS: int = 2
# Per-call retry budget for transient failures (429/5xx/timeout); distinct from
# max_retry_rounds (field re-extraction). A TPM limit is a rolling 60s window, so a
# leaf must outlast it: 60s / ~6s mean backoff ⇒ 10 attempts, each honoring Retry-After.
DEFAULT_MAX_API_RETRIES: int = 10
# Max leaf extraction calls in flight at once. Firing every leaf of a round
# simultaneously overwhelms provider rate limits (429 storms) and, with retries,
# spirals into far more calls. A bounded concurrency window smooths the burst -
# the standard token-bucket / max-concurrency control for batched LLM calls
# (rate-limiting systems, arXiv:2602.11741; LLM batching guidance). Conservative
# default for free tiers; raise it on higher-throughput plans.
DEFAULT_MAX_CONCURRENT_CALLS: int = 4
# Hard cap on fields per leaf (one API call), enforced as a difficulty-weighted load
# (see _reliability_load): ~50 easy fields, fewer hard ones. Cramming many fields into
# one call degrades reliability - instruction-following accuracy falls as the number
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
# Grounding accept threshold in [0, 1]: 0.5 admits exact (1.0) / all-words (0.85) / fuzzy
# (0.7) and rejects partial (0.4) / absent (0.0) - the natural cut in the score ladder.
DEFAULT_GROUNDING_MIN_SCORE: float = 0.5


# ---------------------------------------------------------------------------
# ExtractionConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """Per-call configuration for the nfield extraction pipeline.

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
        reasoning_model: When ``True``, the model is treated as a reasoning model
            and its thinking is disabled on each call so it does not consume the
            answer's output budget. Default ``False``.
        chars_per_token: Override the characters-per-token ratio used to size the
            context-window budget. ``None`` (default) uses a script-aware estimate
            keyed by ``document_language`` (English ~4.0, CJK ~1.5). Set a float to
            pin the exact ratio for a known model. Must be > 0 when set.
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
            ``NEEDS_REVALIDATION``. It is a no-op without
            ``inject_dependencies`` - a dependent is only stale if it consumed
            the upstream value via injection. Default ``False``.
        knowledge_fallback: When ``True``, fields the document does not state may
            be filled from the model's own knowledge instead of left ``NULL``. Best
            for well-known subject matter; risks unsourced values on private
            documents. Default ``False``.
        max_fields_per_call: Per-leaf reliability budget in difficulty-weighted
            units. A leaf grows while ``Σ (1 + λ·D(f)) <= max_fields_per_call``,
            where ``D(f)`` is each field's difficulty: a trivial field costs ~1
            unit, a hard one more. So a leaf holds up to ~``max_fields_per_call``
            easy fields, or fewer hard ones - bounding by reliability, not raw
            count or token budget alone (a large window cannot cram hundreds of
            fields into one unreliable call). Default 50 (the production
            reliability heuristic); forces K = O(load / budget) small leaves.
        max_concurrent_calls: Maximum leaf extraction calls in flight at once.
            Bounds the concurrency of each execution round so a wide schema does
            not fire dozens of calls simultaneously and trip provider rate limits.
            Default 4 (safe for free tiers); raise for higher-throughput plans.
        max_api_retries: Per-call retry budget for transient failures (429 / 5xx /
            timeout), honoring Retry-After. Distinct from ``max_retry_rounds``
            (field re-extraction). Default 10 (outlasts a rolling-window TPM storm);
            lower for fail-fast. Must be > 0.
        ground_values: When ``True``, every filled value of a groundable type (string /
            number / integer) is labelled with how well the excerpt supports it and the
            run reports a ``hallucination_rate``. Non-destructive: a weak label is
            reported, never dropped, because a correct value is often not verbatim (units,
            derived periods). Enum values are ``schema_derived`` (chosen from the schema,
            already validated) and excluded from the metric; booleans and structural types
            are never grounded. Default ``False``.
        grounding_min_score: Minimum grounding score in ``[0, 1]`` for a value to count
            as supported by the source. Only consulted when ``ground_values`` is
            ``True``. Default 0.5.
        provenance: When ``True``, the result carries ``provenance``, a map of each
            value's dot-path to its ``[start, end)`` char offsets in the source
            document (a value located verbatim only). Adds one document scan per value;
            independent of ``ground_values``. Default ``False``.
        fallback_model: Optional stronger model to escalate to. After the recovery pass
            exhausts its retries, any field still failing is re-extracted **once** on
            this model (e.g. a larger model the primary could not satisfy). ``None``
            (default) disables escalation, keeping the run single-model.
        validate_schema: When ``True`` (default), a provably-unsatisfiable schema
            (``minimum > maximum``, empty ``enum``, ``minLength > maxLength``,
            uncompilable ``pattern``, …) raises a ``SchemaError`` before any API call,
            with the field path and a fix hint. The checks are *necessary*
            contradictions, so a valid schema is never rejected. Set ``False`` to skip.
        cache: Response cache. ``False`` (default) makes every model call live;
            ``True`` builds an in-process LRU cache; a ``ResponseCache`` instance
            (e.g. ``DiskCache("path")``) persists across runs or plugs a custom
            backend. Keyed on the exact request (model, messages, output ceiling), so
            an identical call returns the stored text and no different request is ever
            served a cached answer.

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
    # Disable a reasoning model's thinking per call so it does not eat the answer's
    # output budget (see providers/_reasoning.py).
    reasoning_model: bool = False
    # None → script-aware estimate by document_language; a float pins the model's
    # exact ratio (see providers/_token_budget.py).
    chars_per_token: float | None = None
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
    max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS
    max_api_retries: int = DEFAULT_MAX_API_RETRIES
    # When True, validate values exactly as extracted (no lenient normalization of
    # formatted numbers/booleans). Default lenient: accept "$1,234,568" as 1234568.
    strict_validation: bool = False
    # Re-extract conflicting and revalidation-flagged fields during the recovery pass
    # rather than reporting them unresolved.
    recover_conflicts: bool = True
    # Give fields whose Stage 4 call exhausted its transient budget (429 / timeout) one
    # more try in recovery - the rolling-window rate limit has refilled by then. Set False
    # to leave them unrecovered (no extra load on a still-throttled API).
    recover_call_failed: bool = True
    # Reject a provably-unsatisfiable schema (e.g. minimum > maximum, empty enum) before
    # any API call. On by default: the checks are necessary contradictions, so a valid
    # schema is never rejected; set False to skip the preflight entirely.
    validate_schema: bool = True
    # Grounding (anti-hallucination): off by default so behaviour is unchanged. When on,
    # it labels each value's support non-destructively (never drops), since correct
    # values are often not verbatim; enum choices are schema-derived and exempt.
    ground_values: bool = False
    grounding_min_score: float = DEFAULT_GROUNDING_MIN_SCORE
    # Attach source char offsets [start, end) per value to the result (result.provenance).
    # Off by default; adds a document scan per value when on. Independent of grounding.
    provenance: bool = False
    # Stronger model to escalate still-failing fields to after recovery; None disables.
    fallback_model: str | None = None
    # Fill the schema from model knowledge, no document; the prompt answers NULL when
    # unsure (arXiv:2404.10960). Grounding off; reports answer/abstain rates. One call/leaf.
    closed_book: bool = False
    # Opt-in stronger abstention: sample each leaf twice, keep a value only if both agree
    # (arXiv:2602.04853). Doubles calls; no-op unless closed_book is set.
    self_consistency: bool = False
    # Exact-match response cache; False off, True in-memory LRU, or a ResponseCache (see docstring).
    cache: bool | ResponseCache = False

    def __post_init__(self) -> None:
        """Validate settings that have no safe non-positive value.

        Raises:
            ValueError: If ``chars_per_token`` is set to a non-positive ratio.
        """
        if self.chars_per_token is not None and self.chars_per_token <= 0:
            raise ValueError(f"chars_per_token must be > 0, got {self.chars_per_token}")
