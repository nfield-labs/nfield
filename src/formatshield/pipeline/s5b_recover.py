"""Stage 5.5: Missing-Field Recovery Pass (MFRP).

One bounded pass after SFR (Stage 5) and before assembly (Stage 6). Fields still
``EMPTY``/``FAILED`` are recovered without re-touching validated fields:

1. Tree-backtrack — a child of an absent ancestor cannot exist, so it is written
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

import dataclasses
import logging
from typing import TYPE_CHECKING

from formatshield.config import MIN_RECOVERY_FIELDS_PER_CALL
from formatshield.pipeline.s2c_packing import run_stage_2c
from formatshield.pipeline.s3_excerpt import run_stage_3
from formatshield.pipeline.s4_extract import run_stage_4
from formatshield.pipeline.s5_validate import run_stage_5
from formatshield.schema._types import CapacityLeaf, FieldGroup
from formatshield.validation._retry import handle_missing_fields

if TYPE_CHECKING:
    from formatshield.config import ExtractionConfig
    from formatshield.pipeline._state import PipelineState
    from formatshield.providers._protocol import LLMProvider
    from formatshield.schema._types import Field

__all__ = ["run_recovery_pass"]

logger = logging.getLogger(__name__)

_RECOVERY_LEAF_ID: int = -1


def _shrink_budget(config: ExtractionConfig) -> ExtractionConfig:
    """Return a copy of *config* with the reliability budget shrunk for recovery.

    Fields reaching recovery already failed once, so they are re-packed at
    ``recovery_budget_shrink`` of the primary budget (floored at
    ``MIN_RECOVERY_FIELDS_PER_CALL``) — finer decomposition for a more reliable
    retry. A shrink of 1.0 (or larger) is a no-op.

    Args:
        config: The active extraction configuration.

    Returns:
        A config whose ``max_fields_per_call`` is the shrunk recovery budget.
    """
    shrunk = int(config.max_fields_per_call * config.recovery_budget_shrink)
    recovery_cap = max(MIN_RECOVERY_FIELDS_PER_CALL, shrunk)
    recovery_cap = min(recovery_cap, config.max_fields_per_call)
    return dataclasses.replace(config, max_fields_per_call=recovery_cap)


async def run_recovery_pass(
    state: PipelineState,
    provider: LLMProvider,
    config: ExtractionConfig,
) -> PipelineState:
    """Recover still-missing fields in one bounded pass (Stage 5.5).

    Always runs as a core Stage 5 step (architecture engine §5.3); a fast no-op
    when no fields are missing. Validated (``FILLED``) fields are never
    re-extracted.

    Args:
        state: Pipeline state after Stage 5 (blackboard populated).
        provider: LLM provider for the single recovery extraction call.
        config: Extraction configuration (retry rounds for the recovery leaf).

    Returns:
        The same ``PipelineState`` with any recovered fields written to the
        blackboard.

    Example:
        >>> # await run_recovery_pass(state, provider, config)
        >>> callable(run_recovery_pass)
        True
    """
    if state.blackboard is None:
        return state

    bb = state.blackboard
    missing_paths = bb.get_missing() + bb.get_failed()
    if not missing_paths:
        return state

    # 1. Tree-backtrack: a child whose ancestor is itself missing cannot exist, so
    # it is confirmed absent (None) rather than re-queried. Top-level fields have
    # no ancestor and remain eligible for recovery re-extraction.
    missing_set = set(missing_paths)
    orphaned = {p for p in missing_paths if _has_missing_ancestor(p, missing_set)}
    if orphaned:
        backtracked = handle_missing_fields(sorted(orphaned), _all_leaf(state), state.fields)
        for path, value in backtracked.items():
            bb.write_raw(path, value)
    else:
        backtracked = {}

    recover_fields = [
        state.field_by_path[p]
        for p in missing_paths
        if p not in backtracked and p in state.field_by_path
    ]
    if not recover_fields:
        return state

    logger.debug("MFRP: recovering %d missing field(s)", len(recover_fields))

    # 2. Re-pack the missed-only set into capacity-bounded leaves and run it
    #    through Stages 2C-5. Re-packing (not one giant leaf) is essential: a real
    #    document leaves many fields legitimately absent, so a single unsplit
    #    recovery leaf would overflow output and truncate exactly like a mis-packed
    #    primary leaf — re-triggering a per-field retry storm. This mirrors the
    #    architecture engine §5.3 "split_node if the retry node is too large".
    #
    #    Closed-loop adaptive decomposition: these fields already failed once, so
    #    re-pack them FINER than the primary pass (reliability budget shrunk by
    #    recovery_budget_shrink, floored) — smaller, more reliable retry leaves
    #    (MAKER smallest-subtask + error correction, arXiv:2511.09030).
    recovery_config = _shrink_budget(config)
    saved = (state.fields, state.groups, state.leaves, state.execution_order, state.K_min)
    try:
        state.fields = recover_fields
        state.groups = _subgroups_for(recover_fields, state)
        run_stage_2c(state, recovery_config)  # repack missed-only, finer-grained
        run_stage_3(state)
        await run_stage_4(state, provider)
        await run_stage_5(state, provider, config)
    finally:
        state.fields, state.groups, state.leaves, state.execution_order, state.K_min = saved

    return state


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
    (Stage 2C) and excerpt finalisation (Stage 3) reuse the existing BM25 results
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
