"""Stage 6: Assembly.

Zero API calls. Takes the validated blackboard values, assembles them into
nested JSON via the Radix Trie assembler, computes quality metrics, and
builds the final ExtractionResult returned to the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nfield.assembly._quality import compute_quality_score
from nfield.assembly._trie import assemble_json
from nfield.schema._flatten import (
    OPEN_MAP_MARKER,
    UNION_ARRAY_SUFFIX,
    UNION_BASE_MARKER,
)
from nfield.types import ExtractionResult, ExtractionStatus, Metadata

if TYPE_CHECKING:
    from nfield.pipeline._state import PipelineState
    from nfield.schema._types import Field

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
        :class:`~nfield.types.ExtractionResult` ready for the caller.

    """
    assert state.blackboard is not None, "Blackboard must be initialised"

    bb = state.blackboard
    # A union's shadow array field is an internal alternative branch, resolved away
    # before output; it is not one of the schema's fields, so it is not counted.
    fields_total = sum(1 for f in state.fields if not f.path.endswith(UNION_ARRAY_SUFFIX))

    # --- 1. Collect filled values ---
    filled = _resolve_structural_unions(bb.get_filled(), state.fields)
    filled = _fold_open_maps(filled, state.fields)

    # --- 2. Assemble nested JSON ---
    data = assemble_json(filled) if filled else {}

    # --- 3. Quality metrics ---
    report = compute_quality_score(bb, state.fields, state.K, state.K_min)

    # --- 4. Build Metadata ---
    # get_filled() returns only real (non-None) values, so a field the recovery
    # pass confirmed absent (None) is NOT counted as extracted. Everything that is
    # not a real value, a conflict, or pending revalidation is therefore missing -
    # derived as the remainder so the four buckets always sum to fields_total.
    fields_extracted = len(filled)
    fields_conflicted = len(bb.get_conflicts())
    fields_needs_revalidation = len(bb.get_needs_revalidation())
    fields_missing = max(
        0, fields_total - fields_extracted - fields_conflicted - fields_needs_revalidation
    )

    # --- 4a. Grounding metric (only meaningful when grounding ran) ---
    grounded, ungrounded, hallucination_rate = _grounding_metric(state)

    # --- 4b. Closed-book answer/abstain rates (only when closed_book) ---
    answer_rate: float | None = None
    abstain_rate: float | None = None
    if state.closed_book and fields_total:
        answer_rate = fields_extracted / fields_total
        abstain_rate = fields_missing / fields_total

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
        calls_by_origin=dict(state.calls_by_origin),
        fields_grounded=grounded,
        fields_ungrounded=ungrounded,
        hallucination_rate=hallucination_rate,
        unknown_output_lines=state.unknown_lines,
        answer_rate=answer_rate,
        abstain_rate=abstain_rate,
    )

    # --- 5. Determine status ---
    status = _determine_status(fields_extracted, fields_total, fields_missing)

    return ExtractionResult(data=data, metadata=metadata, status=status)


def _grounding_metric(state: PipelineState) -> tuple[int, int, float | None]:
    """Summarise the per-field grounding scores into the run's hallucination metric.

    Counts every grounding-checked value (those Stage 5 scored when grounding was
    enabled) as grounded or ungrounded by the same threshold the gate used, then
    reports the unsupported fraction. A value counts as ungrounded if the source did
    not support it on its best excerpt, even if it was later dropped - that is exactly
    the model's hallucination signal.

    Args:
        state: Pipeline state carrying ``grounding_scores`` and the threshold.

    Returns:
        ``(fields_grounded, fields_ungrounded, hallucination_rate)``; the rate is
        ``None`` when nothing was grounding-checked (grounding off or no groundable
        field).
    """
    scores = state.grounding_scores
    if not scores:
        return 0, 0, None
    threshold = state.grounding_min_score
    grounded = sum(1 for s in scores.values() if s >= threshold)
    ungrounded = len(scores) - grounded
    return grounded, ungrounded, ungrounded / len(scores)


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


def _fold_open_maps(filled: dict[str, Any], fields: list[Field]) -> dict[str, Any]:
    """Fold open-map fields' ``[{key, value}, ...]`` lists back into ``{key: value}`` dicts."""
    open_map_paths = {f.path for f in fields if f.constraints.get(OPEN_MAP_MARKER)}
    if not open_map_paths:
        return filled
    out = dict(filled)
    for path in open_map_paths:
        value = out.get(path)
        # Only fold {key, value} rows; a structural union may have replaced this path
        # with a plain array (the flat branch won), which must pass through untouched.
        if isinstance(value, list) and all(
            isinstance(item, dict) and "key" in item for item in value
        ):
            # Rows with non-string keys are dropped rather than crashing on hashing.
            out[path] = {
                item["key"]: item.get("value")
                for item in value
                if isinstance(item.get("key"), str)
            }
    return out


def _resolve_structural_unions(filled: dict[str, Any], fields: list[Field]) -> dict[str, Any]:
    """Collapse an ``array | object`` anyOf to the branch the document populated.

    Both branches were flattened: the object branch at the base path, the array branch
    at ``base + UNION_ARRAY_SUFFIX``. Whichever came back with rows wins - the object
    branch is preferred when both did, since it is the richer shape. The shadow array
    path is always removed so it never reaches the output.
    """
    bases = {
        f.constraints[UNION_BASE_MARKER] for f in fields if f.constraints.get(UNION_BASE_MARKER)
    }
    if not bases:
        return filled
    out = dict(filled)
    for base in bases:
        shadow = f"{base}{UNION_ARRAY_SUFFIX}"
        object_rows = out.get(base)
        array_items = out.pop(shadow, None)
        object_filled = isinstance(object_rows, list) and len(object_rows) > 0
        array_filled = isinstance(array_items, list) and len(array_items) > 0
        if not object_filled and array_filled:
            out[base] = array_items
    return out
