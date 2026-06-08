"""Stage 6: Assembly.

Zero API calls. Takes the validated blackboard values, assembles them into
nested JSON via the Radix Trie assembler, computes quality metrics, and
builds the final ExtractionResult returned to the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from formatshield.assembly._quality import compute_quality_score
from formatshield.assembly._trie import assemble_json
from formatshield.types import ExtractionResult, ExtractionStatus, Metadata

if TYPE_CHECKING:
    from formatshield.pipeline._state import PipelineState

__all__ = ["run_stage_6"]

# A run is FAILED when more than half its fields are missing, PARTIAL when only
# some are, and SUCCESS when all are present.
_FAILED_MISSING_FRACTION: float = 0.50


def run_stage_6(state: PipelineState) -> ExtractionResult:
    """Assemble extraction result from validated blackboard state.

    Steps:
    1. ``blackboard.get_filled()`` → flat {path: value} dict
    2. ``assemble_json(filled)`` → nested JSON matching schema structure
    3. ``compute_quality_score(...)`` → QualityReport
    4. Build Metadata (includes fields_total from state.fields)
    5. Determine ExtractionStatus (SUCCESS / PARTIAL / FAILED)
    6. Return ExtractionResult

    Args:
        state: Pipeline state from Stage 5 (blackboard fully populated).

    Returns:
        :class:`~formatshield.types.ExtractionResult` ready for the caller.

    Example:
        >>> # result.status is ExtractionStatus.SUCCESS when all fields filled.
        True
    """
    assert state.blackboard is not None, "Blackboard must be initialised"

    bb = state.blackboard
    fields_total = len(state.fields)

    # --- 1. Collect filled values ---
    filled = bb.get_filled()

    # --- 2. Assemble nested JSON ---
    data = assemble_json(filled) if filled else {}

    # --- 3. Quality metrics ---
    report = compute_quality_score(bb, state.fields, state.K, state.K_min)

    # --- 4. Build Metadata ---
    # get_filled() returns only real (non-None) values, so a field the recovery
    # pass confirmed absent (None) is NOT counted as extracted. Everything that is
    # not a real value, a conflict, or pending revalidation is therefore missing —
    # derived as the remainder so the four buckets always sum to fields_total.
    fields_extracted = len(filled)
    fields_conflicted = len(bb.get_conflicts())
    fields_needs_revalidation = len(bb.get_needs_revalidation())
    fields_missing = max(
        0, fields_total - fields_extracted - fields_conflicted - fields_needs_revalidation
    )

    metadata = Metadata(
        K=state.K,
        K_min=state.K_min,
        optimality_gap=report.optimality_gap,
        quality_score=report.quality_score,
        confidence_level=report.confidence_level,
        fields_extracted=fields_extracted,
        fields_total=fields_total,
        fields_missing=fields_missing,
        fields_conflicted=fields_conflicted,
        fields_needs_revalidation=fields_needs_revalidation,
        per_field_confidence=report.per_field_confidence,
        retry_rounds=state.retry_rounds,
        fields_call_failed=len(bb.get_call_failed()),
    )

    # --- 5. Determine status ---
    status = _determine_status(fields_extracted, fields_total, fields_missing)

    return ExtractionResult(data=data, metadata=metadata, status=status)


def _determine_status(
    fields_extracted: int,
    fields_total: int,
    fields_missing: int,
) -> ExtractionStatus:
    """Map field counts to ExtractionStatus.

    Args:
        fields_extracted: Number of successfully filled fields.
        fields_total: Total fields in schema.
        fields_missing: Number of missing/failed fields.

    Returns:
        SUCCESS if all fields filled, FAILED if >50% missing, PARTIAL otherwise.
    """
    if fields_total == 0:
        return ExtractionStatus.SUCCESS
    if fields_extracted == fields_total:
        return ExtractionStatus.SUCCESS
    missing_fraction = fields_missing / fields_total
    if missing_fraction > _FAILED_MISSING_FRACTION:
        return ExtractionStatus.FAILED
    return ExtractionStatus.PARTIAL
