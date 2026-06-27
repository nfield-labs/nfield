"""Quality scoring for extraction results.

Computes aggregate quality metrics from the blackboard state after all
extraction and retry rounds complete:

* ``quality_score``    = fields_extracted / fields_total (fill rate).
* ``confidence_level`` = "HIGH" if every field is filled with no conflicts,
                         "MEDIUM" above an 80% fill rate, "LOW" otherwise.
* per-field confidence = 1.0 (FILLED) | 0.5 (NEEDS_REVALIDATION) | 0.0 (rest).
* ``optimality_gap``   = (K - K_min) / K in [0, 1], lower = better.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nfield.exceptions import AssemblyError

if TYPE_CHECKING:
    from nfield.assembly._blackboard import Blackboard
    from nfield.schema._types import Field

__all__ = [
    "QualityReport",
    "compute_quality_score",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MEDIUM_CONFIDENCE_FILL_THRESHOLD: float = 0.80  # 80% fill rate → MEDIUM
_HIGH_CONFIDENCE_FILL_THRESHOLD: float = 1.0  # 100% fill + no conflicts → HIGH

_CONFIDENCE_FILLED: float = 1.0
_CONFIDENCE_NEEDS_REVALIDATION: float = 0.5
_CONFIDENCE_MISSING_OR_FAILED: float = 0.0


# ---------------------------------------------------------------------------
# QualityReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Quality metrics for a completed extraction run.

    Produced by :func:`compute_quality_score` and consumed by Stage 6
    assembly to populate :class:`~nfield.types.Metadata`.

    Attributes:
        quality_score: Overall fill rate, range [0.0, 1.0]. Higher = better.
        confidence_level: Human-readable tier: ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.
        per_field_confidence: Mapping of field path to confidence score [0.0, 1.0].
        optimality_gap: Fractional gap between actual and minimum API calls.
            Range [0.0, 1.0]; lower = better. ``0.0`` means K == K_min (optimal).
        fields_extracted: Number of fields in FILLED state.
        fields_missing: Number of fields still in EMPTY state after retry.
        fields_conflicted: Number of fields in CONFLICT state.
        fields_needs_revalidation: Number of fields flagged for revalidation.

    Example:
        >>> report = QualityReport(
        ...     quality_score=0.9, confidence_level="MEDIUM",
        ...     per_field_confidence={"name": 1.0, "age": 0.0},
        ...     optimality_gap=0.0, fields_extracted=9, fields_missing=1,
        ...     fields_conflicted=0, fields_needs_revalidation=0,
        ... )
        >>> report.quality_score
        0.9
    """

    quality_score: float
    confidence_level: str
    per_field_confidence: dict[str, float]
    optimality_gap: float
    fields_extracted: int
    fields_missing: int
    fields_conflicted: int
    fields_needs_revalidation: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_quality_score(
    blackboard: Blackboard,
    fields: list[Field],
    K: int,  # noqa: N803 - K is the conventional symbol for the call count
    K_min: int,  # noqa: N803 - K_min is its theoretical lower bound
) -> QualityReport:
    """Compute quality metrics from the blackboard state after all passes.

    Args:
        blackboard: Populated blackboard from Stage 4 + Stage 5.
        fields: All fields from Stage 1 (for total count and path enumeration).
        K: Actual number of API calls made in this extraction run.
        K_min: Theoretical minimum API calls computed by Stage 2C.

    Returns:
        A :class:`QualityReport` summarising extraction quality.

    Example:
        >>> from nfield.assembly._blackboard import Blackboard
        >>> from nfield.schema._types import Field
        >>> bb = Blackboard(["name", "age"])
        >>> bb.write("name", "Alice")
        >>> f_name = Field("name", "string", {}, "", {})
        >>> f_age = Field("age", "integer", {}, "", {})
        >>> report = compute_quality_score(bb, [f_name, f_age], K=2, K_min=1)
        >>> report.fields_extracted
        1
        >>> report.fields_missing
        1
    """
    total = len(fields)
    field_paths = [f.path for f in fields]

    # Count fields by state. get_filled() returns only real (non-None) values, so a
    # confirmed-absent field (recovery writes None) is not counted as extracted.
    # Missing is the remainder, so extracted + missing + conflicted + needs_reval
    # always equals total (and None "confirmed-absent" fields fall into missing).
    filled = blackboard.get_filled()
    conflict_paths = blackboard.get_conflicts()
    needs_rev_paths = blackboard.get_needs_revalidation()

    fields_extracted = len(filled)
    fields_conflicted = len(conflict_paths)
    fields_needs_revalidation = len(needs_rev_paths)
    fields_missing = max(
        0, total - fields_extracted - fields_conflicted - fields_needs_revalidation
    )

    # Quality score: fill rate (real values only)
    quality_score = fields_extracted / total if total > 0 else 0.0

    # Optimality gap: (K - K_min) / K, clamped to [0, 1]
    optimality_gap = _compute_optimality_gap(K, K_min)

    # Per-field confidence
    per_field_confidence = _compute_per_field_confidence(blackboard, field_paths)

    # Confidence level
    confidence_level = _determine_confidence_level(
        fill_rate=quality_score,
        fields_conflicted=fields_conflicted,
        total=total,
    )

    return QualityReport(
        quality_score=quality_score,
        confidence_level=confidence_level,
        per_field_confidence=per_field_confidence,
        optimality_gap=optimality_gap,
        fields_extracted=fields_extracted,
        fields_missing=fields_missing,
        fields_conflicted=fields_conflicted,
        fields_needs_revalidation=fields_needs_revalidation,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_optimality_gap(K: int, K_min: int) -> float:  # noqa: N803
    """Compute the fractional optimality gap.

    ``(K - K_min) / K`` - fraction of extra API calls beyond minimum.
    Range [0.0, 1.0]; 0.0 when K == K_min (optimal); approaches 1.0 as K >> K_min.

    Args:
        K: Actual API calls made.
        K_min: Theoretical minimum API calls.

    Returns:
        Optimality gap in [0.0, 1.0].
    """
    if K <= 0:
        return 0.0
    gap = (K - K_min) / K
    return max(0.0, min(1.0, gap))


def _compute_per_field_confidence(
    blackboard: Blackboard,
    field_paths: list[str],
) -> dict[str, float]:
    """Compute per-field confidence scores from blackboard state.

    - FILLED → 1.0 (validated by type/constraint check)
    - NEEDS_REVALIDATION → 0.5 (found but uncertain)
    - EMPTY / FAILED / CONFLICT → 0.0

    Args:
        blackboard: Populated blackboard.
        field_paths: All field paths to score.

    Returns:
        Mapping of field path to confidence score.
    """
    from nfield.assembly._blackboard import FieldState

    confidence: dict[str, float] = {}
    for path in field_paths:
        try:
            state = blackboard.get_state(path)
        except AssemblyError:
            # Path not registered on the blackboard (should not happen - field_paths
            # come from the same fields it was built with) - score it 0, don't mask
            # other bugs behind a bare except.
            confidence[path] = _CONFIDENCE_MISSING_OR_FAILED
            continue

        if state == FieldState.FILLED:
            confidence[path] = _CONFIDENCE_FILLED
        elif state == FieldState.NEEDS_REVALIDATION:
            confidence[path] = _CONFIDENCE_NEEDS_REVALIDATION
        else:
            confidence[path] = _CONFIDENCE_MISSING_OR_FAILED

    return confidence


def _determine_confidence_level(
    fill_rate: float,
    fields_conflicted: int,
    total: int,
) -> str:
    """Determine the human-readable confidence tier.

    Args:
        fill_rate: Fraction of fields successfully extracted [0.0, 1.0].
        fields_conflicted: Number of fields in CONFLICT state.
        total: Total number of fields.

    Returns:
        ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.
    """
    if total == 0:
        return "HIGH"

    if fill_rate >= _HIGH_CONFIDENCE_FILL_THRESHOLD and fields_conflicted == 0:
        return "HIGH"

    if fill_rate >= _MEDIUM_CONFIDENCE_FILL_THRESHOLD:
        return "MEDIUM"

    return "LOW"
