"""Stage 5.5: Missing-Field Recovery Pass (MFRP).

One bounded pass after SFR (Stage 5) and before assembly (Stage 6). Fields still
``EMPTY``/``FAILED`` are recovered without re-touching validated fields:

1. Tree-backtrack - a child of an absent ancestor cannot exist, so it is written
   ``None`` (confirmed absent) rather than re-queried.
2. The remaining missed-only fields are grouped into a fresh recovery leaf.
3. A targeted excerpt is finalised for that leaf, then it is extracted once and
   validated (reusing Stages 3-5). Exactly one pass, no loops.

Re-extracting only the missing set (never the validated fields) follows the finding
that intrinsic self-correction degrades correct outputs without external signal
(Huang et al., "LLMs Cannot Self-Correct Reasoning Yet", ICLR 2024; Brinkmann et al.,
"Self-Refinement Strategies for LLM Attribute Value Extraction", 2025). Targeted
re-retrieval for the gap set follows missing-information-guided retrieval (MIGRES;
CRAG; FAIR-RAG). The single-pass bound avoids the oscillation that unbounded
self-correction risks.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from nfield.pipeline.s2c_packing import run_stage_2c, safe_excerpt_chars
from nfield.pipeline.s3_excerpt import run_stage_3
from nfield.pipeline.s4_extract import run_stage_4
from nfield.pipeline.s5_validate import run_stage_5
from nfield.retrieval._retarget import record_block_excerpt, targeted_excerpt
from nfield.schema._types import CapacityLeaf, FieldGroup
from nfield.validation._retry import cascade_invalidate, handle_missing_fields

if TYPE_CHECKING:
    from nfield.assembly._blackboard import Blackboard
    from nfield.config import ExtractionConfig
    from nfield.pipeline._state import PipelineState
    from nfield.providers._protocol import LLMProvider
    from nfield.schema._types import Field

__all__ = ["run_recovery_pass"]

logger = logging.getLogger(__name__)

_RECOVERY_LEAF_ID: int = -1
# Recovery re-extracts every failing field once. Repeating the wave multiplies call
# volume and pressures provider rate limits without a commensurate quality gain, so
# the pooled re-extraction runs a single round.
_RECOVERY_ROUNDS: int = 1
# The fallback escalation is a single extra round on the stronger model: one more chance
# for the stragglers, not an unbounded loop (avoids self-correction oscillation).
_FALLBACK_ROUNDS: int = 1


async def run_recovery_pass(
    state: PipelineState,
    provider: LLMProvider,
    config: ExtractionConfig,
    *,
    fallback_provider: LLMProvider | Sequence[LLMProvider] | None = None,
) -> PipelineState:
    """Recover still-missing fields in one bounded pass (Stage 5.5).

    Always runs as a core Stage 5 step; a fast no-op when no fields are missing.
    Validated (``FILLED``) fields are never re-extracted.

    Args:
        state: Pipeline state after Stage 5 (blackboard populated).
        provider: LLM provider for the single recovery extraction call.
        config: Extraction configuration (retry rounds for the recovery leaf).
        fallback_provider: Optional stronger model(s). Any field still failing after
            the primary recovery round is re-extracted once per fallback, in order;
            each model only sees what the previous one left unresolved. A single
            provider behaves as a one-element chain. ``None`` keeps recovery
            single-model.

    Returns:
        The same ``PipelineState`` with any recovered fields written to the
        blackboard.

    Example:
        >>> # await run_recovery_pass(state, provider, config)
    """
    if state.blackboard is None:
        return state
    if fallback_provider is None:
        fallbacks: tuple[LLMProvider, ...] = ()
    elif isinstance(fallback_provider, Sequence):
        fallbacks = tuple(fallback_provider)
    else:
        fallbacks = (fallback_provider,)
    return await _run_consolidated_recovery(state, provider, config, fallbacks)


async def _run_consolidated_recovery(
    state: PipelineState,
    provider: LLMProvider,
    config: ExtractionConfig,
    fallback_providers: tuple[LLMProvider, ...],
) -> PipelineState:
    """Recover every non-filled field in one pooled, bounded retry loop.

    The single retry path for the consolidated configuration. Since validation made
    no API calls, this pass pools all fields that are absent, invalid, conflicting,
    or flagged for revalidation; re-extracts them with a fresh path-aware excerpt and
    a per-field reason; and re-validates - repeating up to ``config.max_retry_rounds``
    times. A child whose ancestor is itself missing is written ``None`` rather than
    re-queried.

    Args:
        state: Pipeline state after validation (blackboard populated).
        provider: LLM provider for recovery extraction calls.
        config: Extraction configuration (retry rounds, conflict handling).

    Returns:
        The same ``PipelineState`` with any recovered values written.
    """
    bb = state.blackboard
    if bb is None:
        return state

    # Pool every recoverable field. A call-failed (429 / timeout) field is included by
    # default (config.recover_call_failed): the rate-limit window has refilled by now, so
    # one more attempt usually lands. With the flag off it is left unrecovered.
    call_failed = set(bb.get_call_failed())
    pool: set[str] = set(bb.get_missing())
    if config.recover_call_failed:
        pool |= set(bb.get_failed())
    else:
        pool |= {p for p in bb.get_failed() if p not in call_failed}
    if config.recover_conflicts:
        pool |= set(bb.get_conflicts())
        pool |= set(bb.get_needs_revalidation())
    # Drop deliberate abstentions; genuine failures (bad cast / dropped line) stay recoverable.
    pool -= state.abstained
    if not pool:
        return state

    # Capture each field's failure reason before reopening clears the stored errors. A
    # call-failed field gets a neutral reason: its Stage 4 call never reached the model,
    # so there is no prior output to correct.
    reasons = {p: _failure_reason(bb, p, transient=p in call_failed) for p in pool}

    # A child whose ancestor is itself missing cannot exist: confirm it absent.
    orphaned = {p for p in pool if _has_missing_ancestor(p, pool)}
    backtracked = (
        handle_missing_fields(sorted(orphaned), _all_leaf(state), state.fields) if orphaned else {}
    )
    for path, value in backtracked.items():
        bb.write_raw(path, value)

    recover_paths = [p for p in sorted(pool) if p not in backtracked and p in state.field_by_path]
    if not recover_paths:
        return state

    # Reopen settled or terminal fields so a recovered value can be written.
    for p in recover_paths:
        bb.reopen_for_retry(p)

    logger.debug("Consolidated recovery: %d field(s)", len(recover_paths))

    saved = (
        state.fields,
        state.groups,
        state.leaves,
        state.execution_order,
        state.K_min,
        dict(state.field_reasons),
    )
    state.in_recovery = True
    try:
        still = [state.field_by_path[p] for p in recover_paths]
        still = await _recover_rounds(
            state, provider, config, bb, still, reasons, rounds=_RECOVERY_ROUNDS
        )
        # Fields retrieval never finds often live at the document's structural
        # boundaries (cover page, preamble, end pages), where their terms are
        # drowned by body-wide repetition; one more round reads those regions
        # directly by position.
        if still:
            still = await _recover_rounds(
                state,
                provider,
                config,
                bb,
                still,
                reasons,
                rounds=1,
                round_offset=_RECOVERY_ROUNDS,
                boundary=True,
            )
        # Escalation: re-extract whatever the primary still could not produce on the
        # fallback chain, once per model in order - only the stragglers pay the higher
        # cost, and each model only sees what the previous one left unresolved.
        for index, fallback in enumerate(fallback_providers):
            if not still:
                break
            logger.debug(
                "Recovery fallback %d: escalating %d field(s) to a stronger model",
                index + 1,
                len(still),
            )
            still = await _recover_rounds(
                state,
                fallback,
                config,
                bb,
                still,
                reasons,
                rounds=_FALLBACK_ROUNDS,
                round_offset=_RECOVERY_ROUNDS + 1 + index,
            )
    finally:
        (
            state.fields,
            state.groups,
            state.leaves,
            state.execution_order,
            state.K_min,
            state.field_reasons,
        ) = saved
        state.in_recovery = False

    # A recovered upstream value may make a dependent that consumed it (via injection)
    # stale; flag those dependents for revalidation. No-op without injection,
    # since dependents were then extracted independently.
    if config.cascade_dependency_invalidation and config.inject_dependencies:
        recovered_now = {p for p in recover_paths if p in bb.get_filled()}
        if recovered_now:
            cascade_invalidate(bb, state.dep_dag, recovered_now)

    # Keep-best: a quality-failed array is restored when its re-extraction came
    # back SMALLER - one recovery call cannot out-collect a windowed sweep, so a
    # shorter list is a partial redo, not a correction. Recovery may replace a
    # value with a fuller one, never degrade one.
    filled_after = bb.get_filled()
    for path, original in state.quality_failed_values.items():
        replacement = filled_after.get(path)
        shrunk = (
            isinstance(original, list)
            and isinstance(replacement, list)
            and len(replacement) < len(original)
        )
        if path not in filled_after or replacement == [] or shrunk:
            if replacement == []:
                bb.mark_failed(path, "re-extraction produced nothing")
            bb.write(path, original)
    state.quality_failed_values.clear()

    return state


async def _recover_rounds(
    state: PipelineState,
    provider: LLMProvider,
    config: ExtractionConfig,
    bb: Blackboard,
    still: list[Field],
    reasons: dict[str, str],
    *,
    rounds: int,
    round_offset: int = 0,
    boundary: bool = False,
) -> list[Field]:
    """Re-extract the *still*-failing fields for up to *rounds* on *provider*.

    Each round re-packs the remaining fields, finalises a fresh path-aware excerpt,
    re-extracts, and re-validates (reusing Stages 2C-5); fields that recover drop out
    of the next round. Shared by the primary recovery loop and the fallback escalation,
    so both use the identical re-extraction path on whichever provider they are given.

    Args:
        state: Pipeline state (mutated: ``fields``/``groups``/``leaves`` for the round).
        provider: LLM provider to re-extract with (primary or fallback).
        config: Extraction configuration.
        bb: The run's blackboard.
        still: Fields to attempt this call (the not-yet-recovered set).
        reasons: ``path -> failure reason`` for the re-extraction prompt.
        rounds: Maximum re-extraction rounds to run.
        round_offset: Added to the round index when updating ``state.retry_rounds`` so
            the fallback round counts beyond the primary rounds. Default 0.
        boundary: Extract against the document's head-and-tail excerpt instead of
            a retrieved one (see :func:`_boundary_excerpt`). Default False.

    Returns:
        The fields still failing after these rounds.
    """
    for round_index in range(rounds):
        if not still:
            break
        state.fields = still
        state.groups = _subgroups_for(still, state)
        state.field_reasons = {f.path: reasons[f.path] for f in still if f.path in reasons}
        run_stage_2c(state, config)
        run_stage_3(state)
        _refresh_excerpts(state)
        if boundary:
            for leaf in state.leaves:
                text = _boundary_excerpt(state, leaf.overhead)
                if text:
                    leaf.document_excerpt = text
        await run_stage_4(state, provider)
        await run_stage_5(state, provider, config)
        state.retry_rounds = max(state.retry_rounds, round_offset + round_index + 1)
        recovered = set(bb.get_filled())
        still = [f for f in still if f.path not in recovered]
        # Refresh each survivor's reason from its LATEST failure: the next round
        # must correct the newest mistake, not repeat the first round's.
        for f in still:
            reasons[f.path] = _failure_reason(bb, f.path)
    return still


def _failure_reason(bb: Blackboard, path: str, *, transient: bool = False) -> str:
    """Describe why *path* needs recovery, for the re-extraction prompt.

    Args:
        bb: The run's blackboard.
        path: The field path whose failure to describe.
        transient: ``True`` when the Stage 4 call itself never completed (429 /
            timeout). The model produced no output to correct, so a neutral
            "extract it fresh" reason is returned instead of a correction prompt.

    Returns:
        A short reason string drawn from the field's current state and error.
    """
    from nfield.assembly._blackboard import FieldState

    if transient:
        # The Stage 4 call never reached the model, so there is no prior output to
        # correct - ask for a fresh extraction.
        return (
            "the previous request did not complete; extract this field from the document, or NULL"
        )

    field_state = bb.get_state(path)
    if field_state == FieldState.CONFLICT:
        candidates = " | ".join(str(c) for c in bb.get_conflict_values(path))
        return (
            f"a previous attempt produced conflicting values ({candidates}); extract "
            "the one the document supports, or NULL"
        )
    if field_state == FieldState.NEEDS_REVALIDATION:
        return "a previous attempt was uncertain; extract the exact value, or NULL"
    error = bb.get_error(path)
    if error:
        # Show the model its own rejected value with the objective error: an external,
        # verifiable signal improves correction (DSPy Assertions, arXiv:2312.13382) where
        # self-critique would not (arXiv:2310.01798).
        prior = bb.get_value(path)
        if prior is not None:
            return f"you previously returned {prior!r}, which failed validation: {error}"
        return f"a previous attempt failed validation: {error}"
    return "a previous attempt did not find this field; re-extract it, or NULL"


def _refresh_excerpts(state: PipelineState) -> None:
    """Replace each recovery leaf's excerpt with a fresh, path-aware one.

    Args:
        state: Pipeline state holding the recovery leaves and document segments.
    """
    for leaf in state.leaves:
        fresh = _recover_excerpt(leaf.fields, state, leaf.overhead)
        if fresh:
            leaf.document_excerpt = fresh


def _boundary_excerpt(state: PipelineState, overhead: float) -> str:
    """The document's head and tail segments, up to the excerpt budget.

    Cover pages and end pages carry the defining facts (parties, dates,
    signatures) that lexical retrieval drowns because their terms recur
    body-wide. Position is the signal, so head and tail segments are taken
    directly, alternating until the budget is spent.
    """
    if not state.segments:
        return ""
    cpt = max(state.chars_per_token, 1.0)
    budget = safe_excerpt_chars(state.C_eff, overhead, state.M_O, cpt)
    segments = sorted(state.segments, key=lambda s: s.start)
    head: list[str] = []
    tail: list[str] = []
    used = 0
    i, j = 0, len(segments) - 1
    from_head = True
    while i <= j:
        seg = segments[i] if from_head else segments[j]
        if used + len(seg.text) > budget and (head or tail):
            break
        (head if from_head else tail).append(seg.text)
        used += len(seg.text)
        if from_head:
            i += 1
        else:
            j -= 1
        from_head = not from_head
    return "\n\n".join(head + tail[::-1])


def _recover_excerpt(fields: list[Field], state: PipelineState, overhead: int) -> str | None:
    """Build a fresh excerpt for *fields*, matching the document's routing path.

    A record document re-routes each field to its own block; a large document
    re-ranks every segment for the field set; a small document keeps the Stage 3
    excerpt (the whole document is already in context), signalled by ``None``.

    Args:
        fields: The fields being recovered.
        state: Pipeline state (segments, indices, calibration).
        overhead: The leaf's fixed prompt overhead in tokens.

    Returns:
        A fresh excerpt, or ``None`` to keep the existing leaf excerpt.
    """
    # Per-leaf excerpt budget B_excerpt, identical to Stage 3 (s3_excerpt): capped
    # so input + output stays within the real window even when the tokenizer emits
    # more tokens than the chars-per-token heuristic predicts.
    cpt = max(state.chars_per_token, 1.0)
    safe_tokens = safe_excerpt_chars(state.C_eff, overhead, state.M_O, cpt) / cpt
    budget = min(state.C_usable - overhead, safe_tokens)
    if state.record_block_segments:
        return record_block_excerpt(
            fields,
            state.record_ordinal,
            state.record_header_segments,
            state.record_block_segments,
            budget_tokens=budget,
            chars_per_token=state.chars_per_token,
        )
    if state.lexical_index is not None and state.segments:
        return (
            targeted_excerpt(
                fields,
                state.lexical_index,
                state.segments,
                budget_tokens=budget,
                chars_per_token=state.chars_per_token,
            )
            or None
        )
    return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _has_missing_ancestor(path: str, missing: set[str]) -> bool:
    """Return whether any dot-path ancestor of *path* is itself missing.

    Args:
        path: Dot-notation field path, e.g. ``"address.city"``.
        missing: The set of all still-missing field paths.

    Returns:
        ``True`` if a strict ancestor of *path* is in *missing* (so the field is
        an orphan that cannot exist); ``False`` for top-level paths.
    """
    parts = path.split(".")
    return any(".".join(parts[:depth]) in missing for depth in range(1, len(parts)))


def _all_leaf(state: PipelineState) -> CapacityLeaf:
    """Return a synthetic leaf holding every field, for ancestor existence checks.

    ``handle_missing_fields`` walks the dot-path tree against a leaf's field set;
    using all fields lets it decide whether each missing field's ancestor exists.

    Args:
        state: Pipeline state.

    Returns:
        A throwaway ``CapacityLeaf`` containing all fields.
    """
    return CapacityLeaf(fields=list(state.fields), groups=[], leaf_id=_RECOVERY_LEAF_ID)


def _subgroups_for(fields: list[Field], state: PipelineState) -> list[FieldGroup]:
    """Rebuild groups restricted to the missed fields, reusing their retrieval.

    Each original group is filtered to just its missed fields, keeping that
    group's ``matched_segments`` / ``segment_scores`` / ``D_cost`` so re-packing
    (Stage 2C) and excerpt finalisation (Stage 3) reuse the existing BMX results
    without a new retrieval index. Re-packing these sub-groups produces correctly
    sized recovery leaves instead of one oversized leaf.

    Args:
        fields: The fields being recovered.
        state: Pipeline state holding the Stage 2A/2.5 groups.

    Returns:
        Filtered sub-groups, one per original group that contains a missed field.
    """
    wanted = {f.path for f in fields}
    subgroups: list[FieldGroup] = []
    for g in state.groups:
        sub_fields = [f for f in g.fields if f.path in wanted]
        if sub_fields:
            subgroups.append(
                FieldGroup(
                    parent_path=g.parent_path,
                    fields=sub_fields,
                    matched_segments=g.matched_segments,
                    segment_scores=g.segment_scores,
                    D_cost=g.D_cost,
                )
            )
    return subgroups
