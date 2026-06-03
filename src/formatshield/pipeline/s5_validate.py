"""Stage 5: Validation & Retry.

Makes R API calls (R ~ 0.104 x K expected). For every leaf, validates
extracted values against type + constraint rules (zero API calls). Failed
fields are retried via SFR (Surgical Field Retry) for up to max_retry_rounds.
After each retry round the recovered values are re-validated.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from formatshield.validation._retry import orchestrate_retry
from formatshield.validation._type_check import validate_field

if TYPE_CHECKING:
    from formatshield.config import ExtractionConfig
    from formatshield.pipeline._state import PipelineState
    from formatshield.providers._protocol import LLMProvider
    from formatshield.schema._types import CapacityLeaf, Field

__all__ = ["run_stage_5"]

logger = logging.getLogger(__name__)


async def run_stage_5(
    state: PipelineState,
    provider: LLMProvider,
    config: ExtractionConfig,
) -> PipelineState:
    """Validate all extracted values; retry failed fields via SFR.

    For each leaf:
    1. Validate every FILLED field against type + constraint rules.
    2. Collect failures (invalid values + EMPTY fields).
    3. If failures: call orchestrate_retry (max config.max_retry_rounds).
    4. Write recovered values back to blackboard; re-validate each one.
    5. Any field still invalid after retry → mark_failed on blackboard.

    Args:
        state: Pipeline state from Stage 4 (blackboard has extracted values).
        provider: LLM provider (used only for retry calls).
        config: Extraction configuration (max_retry_rounds, z_target).

    Returns:
        Updated ``PipelineState``.

    Example:
        >>> # After run_stage_5, blackboard has no invalid FILLED fields.
        True
    """
    assert state.blackboard is not None, "Blackboard must be initialised"

    for leaf in state.leaves:
        await _validate_leaf(leaf, provider, state, config)

    return state


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _validate_leaf(
    leaf: CapacityLeaf,
    provider: LLMProvider,
    state: PipelineState,
    config: ExtractionConfig,
) -> None:
    """Validate a single leaf's fields; retry failures with SFR.

    Args:
        leaf: CapacityLeaf whose fields to validate.
        provider: LLM provider for retry calls.
        state: Pipeline state.
        config: Extraction config.
    """
    from formatshield.assembly._blackboard import FieldState

    bb = state.blackboard
    if bb is None:
        return

    # --- Step 1: collect failures (single get_filled() snapshot, not per field) ---
    filled = bb.get_filled()
    failed_fields: list[Field] = []
    errors: dict[str, str] = {}

    for f in leaf.fields:
        field_state = bb.get_state(f.path)
        if field_state == FieldState.FILLED:
            valid, err = validate_field(filled.get(f.path), f)
            if not valid:
                failed_fields.append(f)
                errors[f.path] = err or "validation failed"
                bb.mark_failed(f.path, errors[f.path])
        elif field_state in (FieldState.EMPTY, FieldState.PENDING):
            # PENDING: Stage 4 ran but SFEP produced no value for this field.
            # Treat the same as EMPTY — needs retry.
            failed_fields.append(f)
            errors[f.path] = "field not extracted"

    if not failed_fields:
        return

    # --- Step 2: SFR retry ---
    # call_counter folds the retry API calls into the run's total K so the
    # reported cost includes retries, not just first-pass extraction.
    retry_calls = [0]
    recovered = await orchestrate_retry(
        failed_fields=failed_fields,
        errors=errors,
        provider=provider,
        leaf=leaf,
        dep_dag=state.dep_dag,
        config=config,
        call_counter=retry_calls,
    )
    state.K += retry_calls[0]
    # Record that a retry phase ran (bounded by config.max_retry_rounds; the
    # exact per-round count is not surfaced by orchestrate_retry).
    state.retry_rounds = max(state.retry_rounds, 1)

    # --- Step 3: write recovered values; re-validate ---
    for path, value in recovered.items():
        field = state.field_by_path.get(path)
        if field is None:
            continue
        valid, err = validate_field(value, field)
        if valid:
            bb.write(path, value)
        else:
            bb.mark_failed(path, err or "retry value still invalid")
            logger.debug("Field %r still invalid after retry: %s", path, err)

    # --- Step 4: mark fields still EMPTY or PENDING after retry as FAILED ---
    for f in failed_fields:
        if f.path not in recovered:
            current = bb.get_state(f.path)
            if current in (FieldState.EMPTY, FieldState.PENDING):
                bb.mark_failed(f.path, "field absent after retry")
