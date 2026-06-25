"""Stage 5: Validation.

Zero API calls. For every leaf, validates extracted values against type and
constraint rules. Filled values that fail are marked ``FAILED``; fields the model
left ``PENDING`` are marked ``FAILED`` too. When grounding is enabled, a filled value
the leaf's excerpt does not support is also marked ``FAILED`` (anti-hallucination), so
the recovery pass re-extracts it. All re-extraction is performed by the recovery pass
(Stage 5.5), so this stage only settles state.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nfield.validation._grounding import grounding_score, is_groundable
from nfield.validation._type_check import validate_field

if TYPE_CHECKING:
    from nfield.config import ExtractionConfig
    from nfield.pipeline._state import PipelineState
    from nfield.providers._protocol import LLMProvider
    from nfield.schema._types import CapacityLeaf

__all__ = ["run_stage_5"]

logger = logging.getLogger(__name__)


async def run_stage_5(
    state: PipelineState,
    provider: LLMProvider,
    config: ExtractionConfig,
) -> PipelineState:
    """Validate all extracted values (no API calls).

    For each leaf, type- and constraint-checks the filled values and settles each
    field's state. Re-extraction of failures is deferred to the recovery pass.

    Args:
        state: Pipeline state from Stage 4 (blackboard has extracted values).
        provider: LLM provider (unused here; kept for stage-signature uniformity).
        config: Extraction configuration (unused here; kept for uniformity).

    Returns:
        Updated ``PipelineState``.

    Example:
        >>> callable(run_stage_5)
        True
    """
    assert state.blackboard is not None, "Blackboard must be initialised"

    for leaf in state.leaves:
        _validate_leaf(leaf, state)

    if state.ground_values:
        _ground_all(state)

    return state


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_leaf(leaf: CapacityLeaf, state: PipelineState) -> None:
    """Validate a leaf's fields without any API call.

    Filled values are type- and constraint-checked; an invalid one is marked
    ``FAILED``. A field left ``PENDING`` (extracted but never returned) is also
    marked ``FAILED`` so the recovery pass treats it as missing. ``EMPTY``,
    ``FAILED``, ``CONFLICT`` and ``NEEDS_REVALIDATION`` fields are left untouched
    for the recovery pass to re-extract.

    Args:
        leaf: The leaf whose fields to validate.
        state: Pipeline state holding the blackboard.
    """
    from nfield.assembly._blackboard import FieldState

    bb = state.blackboard
    if bb is None:
        return
    filled = bb.get_filled()
    for f in leaf.fields:
        field_state = bb.get_state(f.path)
        if field_state == FieldState.FILLED:
            valid, err = validate_field(filled.get(f.path), f)
            if not valid:
                bb.mark_failed(f.path, err or "validation failed")
        elif field_state == FieldState.PENDING:
            bb.mark_failed(f.path, "field not extracted")


def _ground_all(state: PipelineState) -> None:
    """Score each filled, groundable value against the excerpt it was extracted from.

    Runs after type/constraint validation has settled every leaf. For each filled
    value of a groundable type, the support score is taken as the **maximum** over the
    excerpts of all leaves that contain the field (a field split across leaves is
    grounded if any of its excerpts supports it). The score is recorded on
    ``state.grounding_scores`` for the Stage 6 hallucination metric; a value scoring
    below ``state.grounding_min_score`` is marked ``FAILED`` so the recovery pass
    re-extracts it (``EMPTY``/``FAILED``/``None`` and non-groundable types are skipped).

    Args:
        state: Pipeline state (blackboard, leaves, grounding threshold/scores).
    """
    from nfield.assembly._blackboard import FieldState

    bb = state.blackboard
    if bb is None:
        return
    filled = bb.get_filled()

    best: dict[str, float] = {}
    for leaf in state.leaves:
        excerpt = leaf.document_excerpt
        for f in leaf.fields:
            value = filled.get(f.path)
            if value is None or not is_groundable(f, value):
                continue
            score = grounding_score(value, excerpt, f.type)
            if score > best.get(f.path, -1.0):
                best[f.path] = score

    for path, score in best.items():
        state.grounding_scores[path] = score
        if score < state.grounding_min_score and bb.get_state(path) == FieldState.FILLED:
            bb.mark_failed(
                path,
                f"ungrounded: value not supported by the document (score {score:.2f})",
            )
