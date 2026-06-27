"""Stage 4: Extraction.

Makes K API calls - one per CapacityLeaf - in the order defined by
state.execution_order. Leaves in the same round are executed concurrently
via asyncio.gather. Results are written to state.blackboard.

Emergency split: if a provider raises a context-length error, the leaf
is split in half (by field count) and each half is retried as a new call.
This costs one extra API call but keeps the pipeline alive.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from nfield.extraction._papt import select_template
from nfield.extraction._prompt import build_extraction_prompt
from nfield.extraction._sfep import (
    NEEDS_REVALIDATION,
    count_unknown_paths,
    parse_sfep,
    parse_sfep_failures,
)
from nfield.validation._normalize import normalize_value

if TYPE_CHECKING:
    from nfield.pipeline._state import PipelineState
    from nfield.providers._protocol import LLMProvider
    from nfield.schema._types import CapacityLeaf, Field

__all__ = ["run_stage_4"]

logger = logging.getLogger(__name__)

# Keywords that signal a context-length error from the provider. Groq returns
# "Please reduce the length of the messages or completion." on overflow, so
# "reduce the length" must be recognised alongside the OpenAI-style phrasings.
_CONTEXT_ERROR_KEYWORDS: frozenset[str] = frozenset(
    {
        "context_length_exceeded",
        "maximum_context_length",
        "too long",
        "context window",
        "reduce the length",
        "maximum context length",
    }
)


async def run_stage_4(
    state: PipelineState,
    provider: LLMProvider,
) -> PipelineState:
    """Extract all fields by calling the provider K times.

    Iterates ``state.execution_order`` rounds sequentially. Within each
    round, all leaves run concurrently. Results are written to
    ``state.blackboard``.

    Args:
        state: Pipeline state from Stage 3 (leaves have document_excerpt set).
        provider: LLM provider for structured extraction calls.

    Returns:
        Updated ``PipelineState`` with blackboard values written.

    Example:
        >>> callable(run_stage_4)
        True
    """
    assert state.blackboard is not None, "Blackboard must be initialised before Stage 4"

    # Bound concurrency so a wide round does not fire every leaf at once and trip
    # provider rate limits (429 storms). A semaphore caps in-flight calls; leaves
    # within a round still run concurrently, just at most N at a time.
    semaphore = asyncio.Semaphore(max(1, state.max_concurrent_calls))

    async def _bounded(leaf: CapacityLeaf) -> None:
        async with semaphore:
            await _extract_leaf(leaf, provider, state)

    for round_leaves in state.execution_order:
        if not round_leaves:
            continue
        results = await asyncio.gather(
            *[_bounded(leaf) for leaf in round_leaves],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Leaf extraction failed: %s", result)

    return state


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _extract_leaf(
    leaf: CapacityLeaf,
    provider: LLMProvider,
    state: PipelineState,
) -> None:
    """Run one extraction call for a single leaf and write results to blackboard.

    Args:
        leaf: CapacityLeaf with fields and document_excerpt set.
        provider: LLM provider.
        state: Pipeline state (for blackboard access).
    """
    assert state.blackboard is not None
    for f in leaf.fields:
        state.blackboard.mark_pending(f.path)

    if state.closed_book and state.self_consistency:
        await _extract_leaf_self_consistent(leaf, provider, state)
        return

    try:
        raw_text = await _call_provider(leaf, provider, state)
    except Exception as exc:
        err_str = str(exc).lower()
        if any(kw in err_str for kw in _CONTEXT_ERROR_KEYWORDS):
            # Emergency split: leaf is too large, split in half and retry
            logger.warning("Context overflow on leaf %d - emergency split", leaf.leaf_id)
            await _emergency_split(leaf, provider, state)
            return
        # Non-context error: the call itself failed (after provider retries), so the
        # fields are call-failures, not absent - mark transient for honest reporting.
        for f in leaf.fields:
            state.blackboard.mark_failed(f.path, f"provider error: {exc}", transient=True)
        return

    state.unknown_lines += count_unknown_paths(raw_text, leaf.fields)
    extracted = parse_sfep(raw_text, leaf.fields)
    _write_extracted_to_blackboard(extracted, state)
    _mark_cast_failures(raw_text, leaf.fields, extracted, state)
    state.record_calls("extract")


async def _extract_leaf_self_consistent(
    leaf: CapacityLeaf,
    provider: LLMProvider,
    state: PipelineState,
) -> None:
    """Sample the leaf twice and keep only agreed values; abstain (NULL) on disagreement.

    Opt-in stronger closed-book abstention (arXiv:2602.04853). Plain closed-book uses the
    single-pass path.

    Args:
        leaf: The leaf to extract (its excerpt is empty in closed-book mode).
        provider: LLM provider.
        state: Pipeline state (blackboard, closed_book flag).
    """
    assert state.blackboard is not None
    try:
        raw_a = await _call_provider(leaf, provider, state)
        raw_b = await _call_provider(leaf, provider, state)
    except Exception as exc:
        # The call failed after retries - a transient call failure, not absence.
        for f in leaf.fields:
            state.blackboard.mark_failed(f.path, f"provider error: {exc}", transient=True)
        return

    agreed = _self_consistent(parse_sfep(raw_a, leaf.fields), parse_sfep(raw_b, leaf.fields))
    _write_extracted_to_blackboard(agreed, state)
    # Non-agreed fields are abstentions; record for the recovery skip.
    state.abstained.update(f.path for f in leaf.fields if f.path not in agreed)
    state.record_calls("extract")
    state.record_calls("extract")


def _self_consistent(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Keep fields both samples agree on a concrete value for; drop NULL, sentinel, mismatch.

    Args:
        first: Parsed ``{path: value}`` from the first sample.
        second: Parsed ``{path: value}`` from the second sample.

    Returns:
        The agreed ``{path: value}`` subset.
    """
    return {
        path: value
        for path, value in first.items()
        if value is not None and value is not NEEDS_REVALIDATION and second.get(path) == value
    }


async def _call_provider(leaf: CapacityLeaf, provider: LLMProvider, state: PipelineState) -> str:
    """Build prompt and call provider for a single leaf.

    Args:
        leaf: Leaf to extract.
        provider: LLM provider.
        state: Pipeline state (supplies caller system/user prompt context).

    Returns:
        Raw SFEP text from provider.
    """
    template = select_template(leaf.fields, budget_tokens=leaf.safe_output)
    messages = build_extraction_prompt(
        leaf.fields,
        leaf.document_excerpt,
        template,
        instructions=state.instructions,
        dependency_values=_resolved_dependencies(leaf, state),
        knowledge_fallback=state.knowledge_fallback,
        closed_book=state.closed_book,
        field_reasons=state.field_reasons or None,
    )
    return await provider.complete(messages, max_tokens=leaf.safe_output)


def _resolved_dependencies(leaf: CapacityLeaf, state: PipelineState) -> dict[str, object] | None:
    """Collect upstream dependency values to inject into this leaf's prompt.

    Returns the ``{path: value}`` of dependency fields that (a) this leaf's
    fields depend on, (b) live in a different leaf, and (c) are already FILLED
    on the blackboard from an earlier execution round. Returns ``None`` when
    injection is disabled or there is nothing to inject.

    Args:
        leaf: The leaf about to be extracted.
        state: Pipeline state (dep graph + blackboard).
    """
    if not state.inject_dependencies or state.blackboard is None:
        return None
    leaf_paths = {f.path for f in leaf.fields}
    filled = state.blackboard.get_filled()
    resolved: dict[str, object] = {}
    for f in leaf.fields:
        for dep_path in state.dep_dag.get(f.path, set()):
            if dep_path not in leaf_paths and dep_path in filled:
                resolved[dep_path] = filled[dep_path]
    return resolved or None


async def _emergency_split(
    leaf: CapacityLeaf,
    provider: LLMProvider,
    state: PipelineState,
) -> None:
    """Split an oversized leaf in half and extract each half separately.

    Greedy split: first half = fields[:mid], second half = fields[mid:].
    The excerpt is also halved so the retry shrinks both the field count and
    the context - overflow can be driven by either.

    Args:
        leaf: Oversized leaf to split.
        provider: LLM provider.
        state: Pipeline state.
    """
    from nfield.schema._types import CapacityLeaf

    assert state.blackboard is not None
    # Halve the excerpt as well - the overflow may be excerpt-driven.
    half_excerpt = leaf.document_excerpt[: max(1, len(leaf.document_excerpt) // 2)]

    mid = max(1, len(leaf.fields) // 2)
    for chunk_fields in (leaf.fields[:mid], leaf.fields[mid:]):
        if not chunk_fields:
            continue
        split_leaf = CapacityLeaf(
            fields=chunk_fields,
            groups=[],
            document_excerpt=half_excerpt,
            overhead=leaf.overhead,
            safe_output=leaf.safe_output,
            leaf_id=leaf.leaf_id,
        )
        try:
            raw_text = await _call_provider(split_leaf, provider, state)
            extracted = parse_sfep(raw_text, chunk_fields)
            _write_extracted_to_blackboard(extracted, state)
            _mark_cast_failures(raw_text, chunk_fields, extracted, state)
            state.record_calls("emergency_split")
        except Exception as exc:
            logger.warning("Emergency split leaf failed: %s", exc)
            for f in chunk_fields:
                state.blackboard.mark_failed(
                    f.path, f"extraction failed after split: {exc}", transient=True
                )


def _mark_cast_failures(
    raw_text: str,
    fields: list[Field],
    extracted: dict[str, Any],
    state: PipelineState,
) -> None:
    """Mark fields whose emitted value could not be cast as FAILED with the raw text.

    parse_sfep drops an uncastable value, leaving the field PENDING with no record of it.
    Recording the raw string in the failure message lets recovery show the model its own
    rejected output (DSPy Assertions, arXiv:2312.13382). A field that also produced a
    castable value (it is in *extracted*) keeps that value.

    Args:
        raw_text: The leaf's raw SFEP output.
        fields: The fields this call requested.
        extracted: The successfully parsed ``{path: value}`` for this call.
        state: Pipeline state (blackboard, field lookup).
    """
    assert state.blackboard is not None
    for path, raw in parse_sfep_failures(raw_text, fields).items():
        if path in extracted:
            continue
        field = state.field_by_path.get(path)
        type_name = field.type if field is not None else "value"
        state.blackboard.mark_failed(
            path, f"the value {raw!r} could not be read as a valid {type_name}"
        )


def _write_extracted_to_blackboard(
    extracted: dict[str, Any],
    state: PipelineState,
) -> None:
    """Write parse_sfep results to the blackboard.

    NEEDS_REVALIDATION sentinel → mark_needs_revalidation.
    None value → mark_failed (field is missing from document).
    All other values → write normally.

    Args:
        extracted: {path: value} dict from parse_sfep.
        state: Pipeline state with blackboard.
    """
    assert state.blackboard is not None
    for path, value in extracted.items():
        if value is NEEDS_REVALIDATION:
            state.blackboard.mark_needs_revalidation(path)
        elif value is None:
            # Closed-book NULL = abstention; record it so recovery skips it.
            if state.closed_book:
                state.abstained.add(path)
            state.blackboard.mark_failed(path, "field not found in document (LLM output NULL)")
        else:
            # Canonicalize a formatted value to its schema type before storing, so a
            # number field holds a number (schema-valid output) and validation does not
            # reject it on format alone. Lossless-or-decline; skipped in strict mode.
            field = state.field_by_path.get(path)
            if field is not None and not state.strict_validation:
                value = normalize_value(value, field)
            state.blackboard.write(path, value)
