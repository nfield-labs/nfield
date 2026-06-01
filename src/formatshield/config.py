
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
        _logger.warning(
            "Built-in domain config file not found: %s", _DOMAIN_CONFIGS_PATH
        )
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
        f"No DomainConfig registered for domain {domain!r}. "
        f"Available domains: {available}",
        hint=f"Use one of: {', '.join(available)}"
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
        evidence_score_threshold: Minimum BM25/semantic evidence score
            for a chunk to be included in extraction context. Default 0.3.
        use_advanced_sfr: Enable advanced Semantic Field Routing (SFR)
            for improved precision on large schemas. Default ``False``.

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
