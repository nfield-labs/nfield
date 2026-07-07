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
    OPEN_MAP_MERGE_MARKER,
    UNION_ARRAY_SUFFIX,
    UNION_BASE_MARKER,
    UNION_KIND_MARKER,
    WILDCARD_SUFFIX,
)
from nfield.types import ExtractionResult, ExtractionStatus, Metadata
from nfield.validation._grounding import GroundingStatus, find_span

if TYPE_CHECKING:
    from nfield.pipeline._state import PipelineState
    from nfield.schema._types import Field

__all__ = ["run_stage_6"]

# A run is FAILED when more than half its fields are missing, PARTIAL when only
# some are, and SUCCESS when all are present.
_FAILED_MISSING_FRACTION: float = 0.50
# A wildcard field (path + WILDCARD_SUFFIX) assembles under this single dict key.
_WILDCARD_KEY: str = WILDCARD_SUFFIX.lstrip(".")


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
    data = _merge_wildcard_maps(data, state.fields)

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

    provenance = _compute_provenance(state) if state.include_provenance else None

    return ExtractionResult(data=data, metadata=metadata, status=status, provenance=provenance)


def _compute_provenance(state: PipelineState) -> dict[str, list[int]]:
    """Locate each filled value in the source and map its path to ``[start, end)``.

    Searches the whole document, so offsets are document-absolute. Only values found
    verbatim (including numeric and unit renderings) get an entry; non-verbatim values,
    enum choices, and non-groundable types are omitted.

    Args:
        state: Pipeline state (blackboard values, field types, segments).

    Returns:
        A ``{path: [start, end]}`` map; empty if nothing was located.
    """
    bb = state.blackboard
    if bb is None:
        return {}
    doc_text = "\n".join(s.text for s in state.segments)
    spans: dict[str, list[int]] = {}
    for path, value in bb.get_filled().items():
        field = state.field_by_path.get(path)
        if field is None:
            continue
        span = find_span(value, doc_text, field)
        if span is not None:
            spans[path] = [span[0], span[1]]
    return spans


def _grounding_metric(state: PipelineState) -> tuple[int, int, float | None]:
    """Summarise the per-field grounding labels into the run's support metric.

    Counts every grounding-checked value (those Stage 5 labelled when grounding was
    enabled) as supported or unsupported by the configured threshold, then reports the
    unsupported fraction. Schema-derived values (enum choices, validated against the
    allowed set) are excluded: they are not quoted from the prose, so a literal search
    is not a support signal for them. The value is never dropped; this is a reported
    signal, not a gate.

    Args:
        state: Pipeline state carrying ``grounding_results`` and the threshold.

    Returns:
        ``(fields_grounded, fields_ungrounded, hallucination_rate)``; the rate is
        ``None`` when nothing was grounding-checked (grounding off, or only
        schema-derived / non-groundable fields).
    """
    threshold = state.grounding_min_score
    checked = [
        r
        for r in state.grounding_results.values()
        if r.status is not GroundingStatus.SCHEMA_DERIVED
    ]
    if not checked:
        return 0, 0, None
    grounded = sum(1 for r in checked if r.score >= threshold)
    ungrounded = len(checked) - grounded
    return grounded, ungrounded, ungrounded / len(checked)


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

    Both branches were flattened: the object branch (its open-map leaf or fixed-property
    fields), and the array branch at ``base + UNION_ARRAY_SUFFIX``. The object branch wins
    when any of its paths is populated, since it is the richer shape; otherwise the array
    branch moves onto the base. The losing branch's paths are dropped so the base is never
    both a list and a parent, and the shadow array path never reaches the output.
    """
    object_paths: dict[str, list[str]] = {}
    for f in fields:
        base = f.constraints.get(UNION_BASE_MARKER)
        if base and f.constraints.get(UNION_KIND_MARKER) == "object":
            object_paths.setdefault(base, []).append(f.path)
    if not object_paths:
        return filled
    out = dict(filled)
    for base, obj_paths in object_paths.items():
        array_items = out.pop(f"{base}{UNION_ARRAY_SUFFIX}", None)
        object_filled = any(_is_present(out.get(p)) for p in obj_paths)
        array_filled = isinstance(array_items, list) and len(array_items) > 0
        if not object_filled and array_filled:
            for path in obj_paths:
                out.pop(path, None)
            out[base] = array_items
    return out


def _is_present(value: Any) -> bool:
    """True when a value counts as populated - non-empty for containers/strings."""
    if isinstance(value, (list, dict, str)):
        return len(value) > 0
    return value is not None


def _merge_wildcard_maps(data: dict[str, Any], fields: list[Field]) -> dict[str, Any]:
    """Merge an additionalProperties open map's dynamic keys into its parent object.

    The open map is folded to a dict that assembles under a ``*`` key beside the
    object's fixed keys; lift its entries up into the parent and drop the ``*``.
    """
    for f in fields:
        if not f.constraints.get(OPEN_MAP_MERGE_MARKER):
            continue
        parent_path = f.path[: -len(WILDCARD_SUFFIX)] if f.path.endswith(WILDCARD_SUFFIX) else ""
        node = _navigate(data, parent_path)
        if isinstance(node, dict) and isinstance(node.get(_WILDCARD_KEY), dict):
            for key, value in node.pop(_WILDCARD_KEY).items():
                node.setdefault(key, value)
    return data


def _navigate(data: dict[str, Any], dot_path: str) -> Any:
    """Return the nested dict value at *dot_path* (object keys only), or ``None``."""
    if not dot_path:
        return data
    node: Any = data
    for segment in dot_path.split("."):
        if not isinstance(node, dict) or segment not in node:
            return None
        node = node[segment]
    return node
