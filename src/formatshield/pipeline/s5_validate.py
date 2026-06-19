"""Stage 5: Validation.

Zero API calls. For every leaf, validates extracted values against type and
constraint rules. Filled values that fail are marked ``FAILED``; fields the model
left ``PENDING`` are marked ``FAILED`` too. All re-extraction is performed by the
recovery pass (Stage 5.5), so this stage only settles state.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from formatshield.validation._type_check import validate_field

if TYPE_CHECKING:
    from formatshield.config import ExtractionConfig
    from formatshield.pipeline._state import PipelineState
    from formatshield.providers._protocol import LLMProvider
    from formatshield.schema._types import CapacityLeaf

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
    from formatshield.assembly._blackboard import FieldState

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
