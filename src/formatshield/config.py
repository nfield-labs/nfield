"""FormatShield extraction configuration.

This module exposes two primary configuration objects:

* ``DomainConfig`` — domain-specific tuning parameters (token budgets,
  confidence thresholds) loaded from ``_data/domain_configs.json`` at
  import time.
* ``ExtractionConfig`` — per-call extraction settings that control
  chunking ratios, retry rounds, model selection, and more.

A simple domain registry lets callers register custom domain configs
that override the built-in ones.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import SchemaError

__all__ = [
    "DomainConfig",
    "ExtractionConfig",
    "get_domain_config",
    "register_domain",
]

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants — never use magic numbers inline
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT_UTILIZATION_RATIO: float = 0.50
DEFAULT_MAX_RETRY_ROUNDS: int = 2
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
# Hard cap on fields per leaf (one API call). Token budgets alone do not bound a
# call: each field's output is tiny (~a dozen tokens), so a generous context window
# would otherwise cram hundreds of fields into one call — exactly the regime that
# degrades to unreliable output. Production teams cap schemas at ~50 fields because
# beyond it results become unreliable regardless of model (the paper's survival
# heuristic; arXiv:2604.* §"50 fields"). This enforces K = O(N / cap) small,
# reliably-extractable leaves — the core N-field decomposition guarantee.
DEFAULT_MAX_FIELDS_PER_CALL: int = 50
# Opt-in dynamic cap (Stage 0.5): when enabled, the static 50 above is replaced by
# a per-model value MEASURED from a 2-point reliability probe (IFScale-style decay
# fit). target_field_reliability is the single meaningful knob — the per-field
# accuracy SLA the cap is solved for, not a tuned constant.
DEFAULT_CALIBRATE_FIELD_CAP: bool = False
DEFAULT_TARGET_FIELD_RELIABILITY: float = 0.95
DEFAULT_Z_TARGET: float = 1.645
DEFAULT_THINK_PHASE_BUDGET_MIN: int = 100
DEFAULT_THINK_PHASE_BUDGET_MAX: int = 150
DEFAULT_EVIDENCE_SCORE_THRESHOLD: float = 0.3

_DATA_DIR = Path(__file__).parent / "_data"
_DOMAIN_CONFIGS_PATH = _DATA_DIR / "domain_configs.json"

# Required keys for a domain config dict
_DOMAIN_CONFIG_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"domain", "p90_string_tokens", "expected_array_size", "confidence_thresholds"}
)


# ---------------------------------------------------------------------------
# DomainConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DomainConfig:
    """Domain-specific tuning parameters for the extraction pipeline.

    Args:
        domain: Identifier string for the domain, e.g. ``"medical"``.
        p90_string_tokens: 90th-percentile token length for string fields in
            this domain. Used to compute chunk token budgets.
        expected_array_size: Expected number of items in array fields for
            this domain. Used to pre-allocate retrieval passes.
        confidence_thresholds: Mapping of confidence tier label → minimum
            score, e.g. ``{"HIGH": 0.92, "MEDIUM": 0.75}``.

    Example:
        >>> cfg = DomainConfig(
        ...     domain="finance",
        ...     p90_string_tokens=25,
        ...     expected_array_size=4,
        ...     confidence_thresholds={"HIGH": 0.93, "MEDIUM": 0.78},
        ... )
        >>> cfg.domain
        'finance'
    """

    domain: str
    p90_string_tokens: int
    expected_array_size: int
    confidence_thresholds: dict[str, float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainConfig:
        """Construct a ``DomainConfig`` from a raw dictionary.

        Args:
            data: Dictionary that must contain all required keys:
                ``domain``, ``p90_string_tokens``, ``expected_array_size``,
                and ``confidence_thresholds``.

        Returns:
            A new ``DomainConfig`` instance populated from *data*.

        Raises:
            SchemaError: If any required key is absent from *data*.

        Example:
            >>> DomainConfig.from_dict({
            ...     "domain": "legal",
            ...     "p90_string_tokens": 80,
            ...     "expected_array_size": 4,
            ...     "confidence_thresholds": {"HIGH": 0.95, "MEDIUM": 0.80},
            ... })
            DomainConfig(domain='legal', ...)
        """
        missing = _DOMAIN_CONFIG_REQUIRED_KEYS - data.keys()
        if missing:
            raise SchemaError(
                f"DomainConfig dict is missing required keys: {sorted(missing)}",
                hint="Ensure all of domain, p90_string_tokens, expected_array_size, "
                "confidence_thresholds are present.",
            )
        return cls(
            domain=data["domain"],
            p90_string_tokens=data["p90_string_tokens"],
            expected_array_size=data["expected_array_size"],
            confidence_thresholds=dict(data["confidence_thresholds"]),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_domain_configs() -> dict[str, DomainConfig]:
    """Load built-in domain configs from the bundled JSON file.

    Returns:
        Mapping of domain name → ``DomainConfig``. Returns an empty dict
        if the JSON file is missing or cannot be parsed.
    """
    if not _DOMAIN_CONFIGS_PATH.exists():
        _logger.warning("Built-in domain config file not found: %s", _DOMAIN_CONFIGS_PATH)
        return {}
    try:
        raw: list[dict[str, Any]] = json.loads(_DOMAIN_CONFIGS_PATH.read_text(encoding="utf-8"))
        return {entry["domain"]: DomainConfig.from_dict(entry) for entry in raw}
    except Exception as exc:
        _logger.warning("Failed to load domain configs: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Module-level registries
# ---------------------------------------------------------------------------

_builtin_domains: dict[str, DomainConfig] = _load_domain_configs()
_domain_registry: dict[str, DomainConfig] = {}


# ---------------------------------------------------------------------------
# Public registry API
# ---------------------------------------------------------------------------


def register_domain(config: DomainConfig) -> None:
    """Register a custom ``DomainConfig``, overriding any built-in with the same name.

    User-registered configs always take precedence over built-in ones in
    ``get_domain_config``.

    Args:
        config: The ``DomainConfig`` to register.

    Returns:
        None

    Example:
        >>> from formatshield.config import DomainConfig, register_domain, get_domain_config
        >>> custom = DomainConfig(
        ...     domain="biotech",
        ...     p90_string_tokens=60,
        ...     expected_array_size=6,
        ...     confidence_thresholds={"HIGH": 0.94, "MEDIUM": 0.76},
        ... )
        >>> register_domain(custom)
        >>> get_domain_config("biotech").domain
        'biotech'
    """
    _domain_registry[config.domain] = config


def get_domain_config(domain: str) -> DomainConfig:
    """Retrieve a ``DomainConfig`` by domain name.

    User-registered configs take precedence over built-in configs.

    Args:
        domain: The domain identifier string, e.g. ``"medical"``.

    Returns:
        The matching ``DomainConfig``.

    Raises:
        SchemaError: If no config is registered for *domain* in either the
            user registry or the built-in set.

    Example:
        >>> get_domain_config("general").p90_string_tokens
        35
    """
    if domain in _domain_registry:
        return _domain_registry[domain]
    if domain in _builtin_domains:
        return _builtin_domains[domain]
    available = sorted({*_builtin_domains, *_domain_registry})
    raise SchemaError(
        f"No DomainConfig registered for domain {domain!r}. Available domains: {available}",
        hint=f"Use one of: {', '.join(available)}",
    )


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
        calibrate_field_cap: When ``True``, run a one-time per-model reliability
            probe (Stage 0.5) and REPLACE the static ``max_fields_per_call`` with a
            measured value — the largest field count that still meets
            ``target_field_reliability``. Costs two small probe calls per engine
            (cached). Default ``False`` (use the static cap, no extra calls).
        target_field_reliability: The per-field accuracy SLA the dynamic cap is
            solved for (only used when ``calibrate_field_cap`` is ``True``). Higher
            → smaller, safer leaves. Range (0, 1). Default 0.95.
        recovery_budget_shrink: Fraction of the reliability budget used when the
            recovery pass re-packs fields that failed the first attempt. < 1 makes
            recovery decompose FINER (smaller, more reliable leaves) where the
            primary pass struggled — closed-loop "smallest subtask + error
            correction". Default 0.5; floored at ``MIN_RECOVERY_FIELDS_PER_CALL``.
        max_concurrent_calls: Maximum leaf extraction calls in flight at once.
            Bounds the concurrency of each execution round so a wide schema does
            not fire dozens of calls simultaneously and trip provider rate limits.
            Default 4 (safe for free tiers); raise for higher-throughput plans.

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
    calibrate_field_cap: bool = DEFAULT_CALIBRATE_FIELD_CAP
    target_field_reliability: float = DEFAULT_TARGET_FIELD_RELIABILITY
    recovery_budget_shrink: float = DEFAULT_RECOVERY_BUDGET_SHRINK
    max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS
