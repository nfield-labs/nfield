"""Stage 5: Validation & Retry.

Makes R API calls (R ~ 0.104 x K expected). For every leaf, validates
extracted values against type + constraint rules (zero API calls). Failed
fields are retried via SFR (Surgical Field Retry) for up to max_retry_rounds.
After each retry round the recovered values are re-validated.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from formatshield.retrieval._retarget import targeted_excerpt
from formatshield.validation._retry import cascade_invalidate, orchestrate_retry
from formatshield.validation._type_check import validate_field

if TYPE_CHECKING:
    from formatshield.config import ExtractionConfig
    from formatshield.pipeline._state import PipelineState
    from formatshield.providers._protocol import LLMProvider
    from formatshield.schema._types import CapacityLeaf, Field

__all__ = ["run_stage_5"]

logger = logging.getLogger(__name__)

# Floor on the targeted-retry excerpt budget so re-retrieval always has room for
# a few segments even when a leaf's overhead + output nearly fill C_usable.
_MIN_RETRY_EXCERPT_TOKENS: float = 256.0


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
    5. Settle the rest: retry-call failure → real reason; reopened conflict/reval
       → needs-revalidation (human review); otherwise → failed (absent).

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
    # Paths reopened from a settled/terminal state (FAILED is write-able again, but
    # CONFLICT/NEEDS_REVALIDATION are not) — tracked so that if they still do not
    # recover, we surface them for human review instead of a bare "absent".
    reopened: set[str] = set()

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
        elif field_state == FieldState.FAILED:
            # Flaw A: Stage 4 marked this FAILED (usually NULL). Retry it — with
            # fresh re-retrieval below it can still be recovered.
            failed_fields.append(f)
            errors[f.path] = bb.get_error(f.path) or "field not found in document"
        elif field_state == FieldState.CONFLICT:
            # Flaw D: leaves disagreed — adjudicate (show candidates, model picks the
            # grounded one; self-consistency, Wang arXiv:2203.11171). Reopen for write.
            cand_text = " | ".join(str(c) for c in bb.get_conflict_values(f.path))
            errors[f.path] = (
                f"conflicting values were extracted ({cand_text}); choose the one "
                "the document supports, or NULL if none is"
            )
            bb.reopen_for_retry(f.path)
            reopened.add(f.path)
            failed_fields.append(f)
        elif field_state == FieldState.NEEDS_REVALIDATION:
            # Flaw D: model was uncertain — re-extract with fresh evidence.
            errors[f.path] = "previously uncertain — re-extract the exact value, or NULL"
            bb.reopen_for_retry(f.path)
            reopened.add(f.path)
            failed_fields.append(f)

    if not failed_fields:
        return

    # Flaw B: re-retrieve a fresh excerpt for the failed fields, so the retry sees
    # different text than the trimmed context that already missed. No-op for small docs.
    retry_excerpt: str | None = None
    if state.bm25_index is not None and state.segments:
        budget = max(_MIN_RETRY_EXCERPT_TOKENS, state.C_usable - leaf.overhead - leaf.safe_output)
        retry_excerpt = (
            targeted_excerpt(
                failed_fields,
                state.bm25_index,
                state.segments,
                budget_tokens=budget,
                chars_per_token=state.chars_per_token,
            )
            or None
        )

    # --- Step 2: SFR retry ---
    # call_counter folds the retry API calls into the run's total K so the
    # reported cost includes retries, not just first-pass extraction. rounds_counter
    # reports the true number of rounds run (flaw C); call_failures records fields
    # whose retry call itself errored (flaw E).
    retry_calls = [0]
    rounds_used = [0]
    call_failures: dict[str, str] = {}
    recovered = await orchestrate_retry(
        failed_fields=failed_fields,
        errors=errors,
        provider=provider,
        leaf=leaf,
        dep_dag=state.dep_dag,
        config=config,
        call_counter=retry_calls,
        rounds_counter=rounds_used,
        call_failures=call_failures,
        system_prompt=state.system_prompt,
        user_prompt=state.user_prompt,
        knowledge_fallback=state.knowledge_fallback,
        retry_excerpt=retry_excerpt,
    )
    state.K += retry_calls[0]
    state.retry_rounds = max(state.retry_rounds, rounds_used[0])

    # --- Step 3: write recovered values; re-validate ---
    recovered_valid: list[str] = []
    for path, value in recovered.items():
        field = state.field_by_path.get(path)
        if field is None:
            continue
        valid, err = validate_field(value, field)
        if valid:
            bb.write(path, value)
            recovered_valid.append(path)
        else:
            bb.mark_failed(path, err or "retry value still invalid")
            logger.debug("Field %r still invalid after retry: %s", path, err)

    # --- Step 3b: CADTR — a recovered upstream value may make dependents stale ---
    # Only meaningful when injection is also on: a dependent is stale only if it
    # actually consumed the upstream value, which happens via dependency injection.
    # Without injection, dependents were extracted independently, so cascading
    # would wrongly discard good values.
    if config.cascade_dependency_invalidation and config.inject_dependencies and recovered_valid:
        invalidated = cascade_invalidate(bb, state.dep_dag, set(recovered_valid))
        if invalidated:
            logger.debug("CADTR flagged %d dependent field(s) for revalidation", len(invalidated))

    # --- Step 4: settle fields that did not recover ---
    # call failed → real reason; was conflict/reval → human review; else → absent.
    for f in failed_fields:
        if f.path in recovered:
            continue
        current = bb.get_state(f.path)
        if current not in (FieldState.EMPTY, FieldState.PENDING):
            continue
        if f.path in call_failures:
            bb.mark_failed(f.path, call_failures[f.path])
        elif f.path in reopened:
            bb.mark_needs_revalidation(f.path)
        else:
            bb.mark_failed(f.path, "field absent after retry")
