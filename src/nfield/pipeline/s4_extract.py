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
import itertools
import json
import logging
import math
import re
from typing import TYPE_CHECKING, Any

from nfield.extraction._papt import dimension_axes, select_template
from nfield.extraction._prompt import build_extraction_prompt
from nfield.extraction._sfep import (
    NEEDS_REVALIDATION,
    count_unknown_paths,
    parse_sfep,
    parse_sfep_failures,
    truncated_json_arrays,
    unclean_json_arrays,
)
from nfield.pipeline.s2c_packing import _FALLBACK_CHARS_PER_TOKEN, safe_excerpt_chars
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Extra resamples for a leaf whose array value needed repair.
_MAX_ARRAY_RESAMPLES: int = 2
# Each recovery halves the excerpt; three levels resolve any near-limit overflow.
_MAX_SPLIT_DEPTH: int = 3
# Tokenizer floor: a reading below this is a spurious usage report, ignored.
_MIN_CALIBRATED_CPT: float = 0.5

# --- Unbounded-array window sweep ---
# Document-sized shared budget: one content pass plus one overlap pass, so it
# neither starves a huge document nor lets many arrays multiply the call count.
_CONTINUATION_DOC_COVER_FACTOR: float = 2.0
# Window bounded by output budget too: items are copied verbatim, so the emitted
# text is roughly the window text; the factor leaves room for JSON structure.
_OUTPUT_WINDOW_FACTOR: float = 0.5
# Consecutive near-empty windows tolerated before stopping. Patience must exceed a
# typical prose gap between item clusters so the tail cluster is still reached.
_CONTINUATION_STOP_AFTER_EMPTY: int = 6
_CONTINUATION_STOP_WHILE_EMPTY: int = 8
_CONTINUATION_SCARCE_ITEMS: int = 3
_CONTINUATION_MIN_YIELD: int = 3
# Output reservation covers item text plus JSON structure; exact window_chars/cpt
# truncates every dense window's tail, so add this structural headroom.
_WINDOW_OUTPUT_HEADROOM: float = 1.25
# Top-relevance windows re-asked for a single array left empty beside yielding siblings.
_FOCUSED_REASK_WINDOWS: int = 2
# Placing a sibling's row in a window: leaves shorter than this are too generic to
# match, and a window must hold this many of one row's leaves to count as its source.
_DECONFLICT_LEAF_MIN_CHARS: int = 4
_DECONFLICT_MIN_LEAF_MATCHES: int = 2
# A document states its global identity once, at the top (what it is, what period or
# scope it covers); a bare mid-document window forces the model to re-infer those
# facts per row. Each window lends at most this fraction to that head context, so
# content keeps the rest.
_PREAMBLE_WINDOW_FRACTION: float = 0.15
# At most this many items all restating the field key = the document's mention of
# the field, not its entries.
_PLACEHOLDER_MAX_ITEMS: int = 2

# --- Array item grounding and dedupe ---
# Fraction of items reappearing bracketed in the excerpt to call an array reference labels.
_LABEL_ARRAY_MIN_FRACTION: float = 0.8
_LABEL_ENTRY_MAX_CHARS: int = 600
# Items this long ground reliably by substring; shorter ones are too generic.
_GROUNDABLE_ITEM_MIN_CHARS: int = 40
_MAX_UNGROUNDED_FRACTION: float = 0.3
# In a long-text list, a bare number is a marker, not an entry.
_LONG_TEXT_ITEM_CHARS: int = 40
# Fraction of adjacent pairs that must ascend by one for a list to be a numbering run.
_ORDINAL_RUN_MIN_FRACTION: float = 0.8
# Trailing items tried as the continuation anchor, and the longest leaves per item.
_ANCHOR_ITEM_TRIES: int = 8
_ANCHOR_ITEM_MIN_CHARS: int = 20
_ANCHOR_LEAVES_PER_ITEM: int = 3
_WS = re.compile(r"\s+")
# Punctuation a model normalises while copying; folded on both sides for substring match.
_GROUND_FOLD = str.maketrans(
    {
        0x2010: "-",
        0x2011: "-",
        0x2012: "-",
        0x2013: "-",
        0x2014: "-",
        0x2018: "'",
        0x2019: "'",
        0x201C: '"',
        0x201D: '"',
    }
)


def _min_continuation_chars(leaf: CapacityLeaf, state: PipelineState) -> int:
    """Character floor below which an uncovered remainder is not worth a sweep call.

    Derived, not asserted: a call's fixed cost is its own prompt overhead, so the
    remainder must carry at least as many content tokens as that overhead - otherwise
    the call is mostly format and cannot pay for itself. Overhead is measured and the
    chars-per-token is calibrated, so the floor holds for any schema, model, or
    tokenizer with no magic number.
    """
    return int(leaf.overhead * max(state.chars_per_token, 1.0))


def _max_continuation_windows_per_doc(leaf: CapacityLeaf, state: PipelineState) -> int:
    """Shared cap on continuation calls for the whole document, sized to the document.

    A document of ``D`` characters is worth ``ceil(D / one output window) *
    factor`` continuation windows across all its arrays combined. The reference
    window uses the fallback density (not the live-calibrated one) so the COUNT is
    stable under calibration - calibration changes a window's CHARS, not how many
    windows a document is worth - which keeps a many-array schema's call count bounded.
    """
    ref_window = max(1.0, leaf.safe_output * _OUTPUT_WINDOW_FACTOR * _FALLBACK_CHARS_PER_TOKEN)
    doc_chars = sum(len(s.text) for s in state.segments)
    windows_to_cover = math.ceil(doc_chars / ref_window)
    return max(1, int(windows_to_cover * _CONTINUATION_DOC_COVER_FACTOR))


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
    *,
    split_depth: int = 0,
) -> None:
    """Run one extraction call for a single leaf and write results to blackboard.

    Args:
        leaf: CapacityLeaf with fields and document_excerpt set.
        provider: LLM provider.
        state: Pipeline state (for blackboard access).
        split_depth: Emergency-split recursion depth of this leaf.
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
        if any(kw in err_str for kw in _CONTEXT_ERROR_KEYWORDS) and split_depth < _MAX_SPLIT_DEPTH:
            # Emergency split: leaf is too large, split in half and retry
            logger.warning("Context overflow on leaf %d - emergency split", leaf.leaf_id)
            await _emergency_split(leaf, provider, state, split_depth=split_depth)
            return
        # Non-context error: the call itself failed (after provider retries), so the
        # fields are call-failures, not absent - mark transient for honest reporting.
        for f in leaf.fields:
            state.blackboard.mark_failed(f.path, f"provider error: {exc}", transient=True)
        return

    state.unknown_lines += count_unknown_paths(raw_text, leaf.fields)
    extracted = parse_sfep(raw_text, leaf.fields)
    state.record_calls("extract" if split_depth == 0 else "emergency_split")
    if not state.in_recovery:
        # An array cut at the output token limit is deterministic - a re-sample
        # repeats the cut - so those paths skip the resample and continue below.
        truncated = truncated_json_arrays(raw_text, leaf.fields)
        # An array that needed JSON repair may be partial; re-sample and keep the best.
        await _reparse_unclean_arrays(raw_text, leaf, provider, state, extracted, skip=truncated)
        # Items that are the document's own bracketed labels are references to
        # entries, not the entries; one corrective retry asks for the full text.
        await _expand_label_arrays(leaf, provider, state, extracted)
        # An array the output limit cut mid-list keeps its salvaged items; the
        # remaining entries are recovered from the document after the last one.
        await _continue_truncated_arrays(leaf, provider, state, extracted, truncated)
        # An unbounded array's items may continue past the excerpt into document
        # regions this call never saw; sweep those regions and extend the arrays.
        await _extend_arrays_over_windows(leaf, provider, state, extracted)
    _write_extracted_to_blackboard(extracted, state)
    _mark_cast_failures(raw_text, leaf.fields, extracted, state)


async def _reparse_unclean_arrays(
    raw_text: str,
    leaf: CapacityLeaf,
    provider: LLMProvider,
    state: PipelineState,
    extracted: dict[str, Any],
    *,
    skip: set[str] | None = None,
) -> None:
    """Re-sample the leaf when an array value needed JSON repair; adopt the first clean one.

    For each list-leaf array whose emission did not parse as valid JSON, take up to
    :data:`_MAX_ARRAY_RESAMPLES` more samples and adopt the first that parses cleanly
    (falling back to the fullest otherwise) - turning an intermittent malformed emission
    into a reliable one at the cost of a few extra calls only when needed. Paths in
    *skip* (output-truncated: a re-sample repeats the cut) are left to continuation.
    """
    pending = unclean_json_arrays(raw_text, leaf.fields) - (skip or set())
    if not pending:
        return
    for _ in range(_MAX_ARRAY_RESAMPLES):
        try:
            raw_n = await _call_provider(leaf, provider, state)
        except Exception as exc:
            logger.warning("Array re-sample failed on leaf %d: %s", leaf.leaf_id, exc)
            return
        state.record_calls("array_resample")
        extracted_n = parse_sfep(raw_n, leaf.fields)
        unclean_n = unclean_json_arrays(raw_n, leaf.fields)
        still_pending: set[str] = set()
        for path in pending:
            new_value = extracted_n.get(path)
            if not isinstance(new_value, list):
                still_pending.add(path)
                continue
            old_value = extracted.get(path)
            old_rows = len(old_value) if isinstance(old_value, list) else -1
            if path not in unclean_n and (new_value or old_rows <= 0):
                # Adopt the clean sample, unless empty while repair recovered rows.
                extracted[path] = new_value
                continue
            if len(new_value) > old_rows:
                extracted[path] = new_value  # still unclean; keep the fullest so far
            still_pending.add(path)
        pending = still_pending
        if not pending:
            return


async def _expand_label_arrays(
    leaf: CapacityLeaf,
    provider: LLMProvider,
    state: PipelineState,
    extracted: dict[str, Any],
) -> None:
    """Retry once when a scalar array came back as document-internal labels.

    A keyed list renders each entry as ``[Label] full text``; a model sometimes
    emits only the labels. Detection is deterministic - the items reappear
    bracketed in the excerpt - and the single corrective call names the mistake.
    The retry is adopted only when its items stop looking like labels.
    """
    suspect = {
        f.path
        for f in leaf.fields
        if isinstance(extracted.get(f.path), list)
        and _looks_like_label_array(extracted[f.path], leaf.document_excerpt)
    }
    if not suspect:
        return
    reasons = dict.fromkeys(
        suspect,
        "these items are the document's bracketed reference labels, not the "
        "entries themselves; output each entry's FULL text as written, from "
        "after its [label] to where the next entry begins",
    )
    try:
        raw = await _call_provider(leaf, provider, state, field_reasons=reasons)
    except Exception as exc:
        logger.warning("Label-array retry failed on leaf %d: %s", leaf.leaf_id, exc)
        return
    state.record_calls("array_resample")
    parsed = parse_sfep(raw, leaf.fields)
    for path in suspect:
        new_value = parsed.get(path)
        if (
            isinstance(new_value, list)
            and new_value
            and not _looks_like_label_array(new_value, leaf.document_excerpt)
        ):
            extracted[path] = new_value
            continue
        # The model insists on labels: resolve each one against the document
        # deterministically - the entry is the text after its bracketed label.
        base = extracted.get(path)
        if isinstance(base, list):
            extracted[path] = _dereference_labels(base, leaf.document_excerpt)


def _dereference_labels(items: list[Any], excerpt: str) -> list[Any]:
    """Replace each bracketed-label item with the entry text that follows its label.

    A keyed list renders ``[Label] entry text ... [NextLabel]``; the entry is the
    text between a label's LAST occurrence (the list itself, not an in-text
    mention) and the next entry's label or ordinal. Items whose label is not in
    the excerpt are kept unchanged.
    """
    next_entry = re.compile(r"\n\s*\d{1,4}\.\s*\[|\n\s*\[[A-Za-z]")
    out: list[Any] = []
    for item in items:
        if not isinstance(item, str):
            out.append(item)
            continue
        label = f"[{_bare_label(item)}]"
        at = excerpt.rfind(label)
        if at == -1:
            out.append(item)
            continue
        start = at + len(label)
        tail = excerpt[start : start + _LABEL_ENTRY_MAX_CHARS]
        cut = next_entry.search(tail)
        entry = (tail[: cut.start()] if cut else tail).strip()
        out.append(entry if entry else _bare_label(item))
    return out


def _array_quality_error(items: list[Any], excerpt: str) -> str | None:
    """Reason an extracted array's items are document furniture, else ``None``.

    Two deterministic failure classes: items that reappear as bracketed labels in
    the excerpt (references to entries, not entries) and arrays dominated by bare
    consecutive integers (list ordinals). Validation uses this to fail a bad
    array so recovery re-extracts it with the reason.
    """
    if _looks_like_label_array(items, excerpt):
        return (
            "the items are the document's bracketed reference labels, not the "
            "entries themselves; output each entry's full text as written"
        )
    strings = [i for i in items if isinstance(i, str) and i.strip()]
    if len(strings) >= 3:
        digits = [s for s in strings if s.strip().isdigit() and len(s.strip()) <= 5]
        if len(digits) > 0.6 * len(strings):
            return (
                "the items are bare list numbers, not the entries' text; output "
                "each entry's full text as written"
            )
        # Grounding: a verbatim item must appear in the document. Items the model
        # reworded, merged, or invented fail the substring check en masse.
        long_items = [s for s in strings if len(s) >= _GROUNDABLE_ITEM_MIN_CHARS]
        if len(long_items) >= 3:
            hay = _ground_norm(excerpt)
            missing = sum(1 for s in long_items if _ground_norm(s) not in hay)
            if missing / len(long_items) > _MAX_UNGROUNDED_FRACTION:
                return (
                    "many items do not match the document text; copy each entry's "
                    "text exactly as written, without rewording or combining entries"
                )
    return None


def _ground_norm(text: str) -> str:
    """Casefold, fold punctuation variants, and collapse whitespace for substring match."""
    return _WS.sub(" ", text.translate(_GROUND_FOLD)).casefold()


def _is_scarce(value: Any) -> bool:
    """True when *value* holds too few items to distinguish from empty."""
    return not isinstance(value, list) or len(value) < _CONTINUATION_SCARCE_ITEMS


def _restates_path_key(item: str, path: str) -> bool:
    """True when *item*'s words include every word of the field's own key.

    "THE LENDERS FROM TIME TO TIME PARTY HERETO" names the ``lenders`` field
    rather than any lender; a real entry does not carry its own field name.
    A trailing ``s`` folds on both sides so a plural key matches its singular
    mention ("TRANCHE A LENDER" for ``lenders``).
    """

    def stem(w: str) -> str:
        return w[:-1] if w.endswith("s") else w

    key_words = {stem(w) for w in path.rsplit(".", 1)[-1].casefold().split("_")}
    words = {stem(w) for w in re.findall(r"[a-z]+", item.casefold())}
    return bool(key_words) and key_words <= words


def _looks_like_label_array(items: list[Any], excerpt: str) -> bool:
    """True when most items reappear as bracketed labels in the excerpt."""
    strings = [item for item in items if isinstance(item, str) and item.strip()]
    if len(strings) < 3:
        return False
    labelled = sum(1 for item in strings if f"[{_bare_label(item)}]" in excerpt)
    return labelled / len(strings) >= _LABEL_ARRAY_MIN_FRACTION


def _bare_label(item: str) -> str:
    """The item with one surrounding bracket pair removed (``[Key]`` -> ``Key``)."""
    text = item.strip()
    if text.startswith("[") and text.endswith("]"):
        return text[1:-1].strip()
    return text


async def _extend_arrays_over_windows(
    leaf: CapacityLeaf,
    provider: LLMProvider,
    state: PipelineState,
    extracted: dict[str, Any],
) -> None:
    """Extend unbounded arrays with items found beyond the leaf's excerpt.

    An unbounded array's item count scales with the document, so on a document
    larger than one context window the excerpt necessarily misses items (a
    bibliography or table continuing for hundreds of pages). The uncovered
    segments are swept in document order, one excerpt-sized window per call,
    asking only for the leaf's unbounded array fields; parsed items are appended
    with exact-duplicate suppression.

    An unbounded array that came back EMPTY is re-collected over the WHOLE
    document, not just the uncovered remainder: a single whole-document call
    under-emits a large array (the model abstains or stops early), so a dedicated
    per-window pass finds the items the combined call skipped. When every array
    already has items, only the region beyond the excerpt is swept.

    A DIMENSION array (its items carry a categorical enum axis, e.g. one value
    reported per category of that axis) is also swept over the whole document even
    when partially filled: its per-category rows are scattered across tables the
    primary pass never gathered into one array, so the uncovered-remainder sweep alone
    misses the rows that sit inside the already-covered excerpt.
    """
    array_fields = [f for f in leaf.fields if _is_unbounded_list_leaf(f)]
    if not array_fields or not state.segments:
        return
    # A tiny array made only of the field's own placeholder phrase ("the <key>
    # party hereto") is not filled; clear it so the sweep rebuilds the real list.
    for f in array_fields:
        items = extracted.get(f.path)
        if (
            isinstance(items, list)
            and 0 < len(items) <= _PLACEHOLDER_MAX_ITEMS
            and all(isinstance(x, str) and _restates_path_key(x, f.path) for x in items)
        ):
            extracted[f.path] = []
    # A scarce array is treated as empty: a few items say nothing about the rest,
    # so a remainder-only region would skip where the missing entries live.
    scarce = any(_is_scarce(extracted.get(f.path)) for f in array_fields)
    dimensioned = any(dimension_axes(f) for f in array_fields)
    if scarce or dimensioned:
        region = sorted(state.segments, key=lambda s: s.start)
    else:
        region = sorted(
            (s for s in state.segments if s.segment_id not in leaf.excerpt_segment_ids),
            key=lambda s: s.start,
        )
    if sum(len(s.text) for s in region) < _min_continuation_chars(leaf, state):
        return

    # Sweep highest-relevance windows first: items cluster where the fields' terms
    # are densest, so relevance order reaches them without scanning the whole document.
    score_by_id: dict[int, float] = {}
    for g in leaf.groups:
        for seg, sc in zip(g.matched_segments, g.segment_scores, strict=False):
            score_by_id[seg.segment_id] = max(score_by_id.get(seg.segment_id, 0.0), sc)
    window_chars = _window_chars(leaf, state)
    preamble = _document_preamble(state, _preamble_cap(leaf, state, window_chars))
    windows = _pack_windows(region, window_chars, score_by_id)
    visit_order = sorted(range(len(windows)), key=lambda i: windows[i][1], reverse=True)
    await _sweep_array_windows(
        [text for text, _ in windows],
        array_fields,
        leaf,
        provider,
        state,
        extracted,
        visit_order=visit_order,
        preamble=preamble,
    )

    # Under output pressure a window fills some arrays and leaves siblings empty; a
    # field still empty beside a yielding sibling is re-asked alone. All-empty = absent.
    still_empty = [f for f in array_fields if not extracted.get(f.path)]
    if still_empty and len(still_empty) < len(array_fields):
        top = [windows[i][0] for i in visit_order[:_FOCUSED_REASK_WINDOWS]]
        for f in still_empty:
            await _sweep_array_windows(
                top, [f], leaf, provider, state, extracted, preamble=preamble
            )

    # A dimension array lacking an axis value a sibling's rows prove the document
    # reports (same axis, same allowed values) was starved by shared output, not
    # absence. All starved arrays are re-asked in ONE pass - together they face far
    # less output competition than the full sweep that starved them, and one call
    # per window replaces one call per field per window. No fallback: when the proof
    # rows locate nowhere, there is no target worth a call.
    window_texts = [text for text, _ in windows]
    starved_fields: list[Field] = []
    starved_reasons: dict[str, str] = {}
    target_union: list[int] = []
    for f, axis, missing, proof in _axis_starved_fields(array_fields, extracted):
        targets = _windows_holding_rows(window_texts, proof)
        if not targets:
            continue
        starved_fields.append(f)
        starved_reasons[f.path] = (
            _continuation_reason(f)
            + f". Related fields show this document reports {axis} values "
            + f"[{', '.join(sorted(missing))}]; this field's own value may be reported "
            + "for those as well - emit an entry for each such figure shown here"
        )
        target_union.extend(i for i in targets if i not in target_union)
    if starved_fields:
        target_union = target_union[: _FOCUSED_REASK_WINDOWS * 2]
        logger.info(
            "axis-starved arrays %s: re-asking together over %d window(s)",
            sorted(f.path for f in starved_fields),
            len(target_union),
        )
        await _sweep_array_windows(
            [window_texts[i] for i in target_union],
            starved_fields,
            leaf,
            provider,
            state,
            extracted,
            preamble=preamble,
            reasons=starved_reasons,
            cap_bonus=len(target_union),
        )


def _axis_starved_fields(
    array_fields: list[Field], extracted: dict[str, Any]
) -> list[tuple[Field, str, set[str], list[Any]]]:
    """Dimension arrays missing an axis value a sibling's rows prove is reported.

    Two arrays sharing an axis definition (same property name, same allowed values)
    describe the same labelled breakdown; when one holds rows for an axis value the
    other lacks entirely, the document demonstrably reports that breakdown, so the
    lack is under-emission rather than absence. Empty arrays are excluded - they are
    the focused re-ask's case, and an axis read from zero rows proves nothing.

    Returns:
        ``(field, axis name, missing values, sibling proof rows)`` per starved array.
    """
    out: list[tuple[Field, str, set[str], list[Any]]] = []
    for f in array_fields:
        mine_items = extracted.get(f.path)
        if not isinstance(mine_items, list) or not mine_items:
            continue
        for axis, allowed in dimension_axes(f):
            allowed_key = (axis, tuple(allowed))
            mine = _axis_values(mine_items, axis)
            missing: set[str] = set()
            proof: list[Any] = []
            for g in array_fields:
                if g is f or (axis, tuple(dict(dimension_axes(g)).get(axis, ()))) != allowed_key:
                    continue
                sib_items = extracted.get(g.path)
                if not isinstance(sib_items, list):
                    continue
                for item in sib_items:
                    if not isinstance(item, dict):
                        continue
                    value = str(item.get(axis))
                    if value in allowed and value not in mine:
                        missing.add(value)
                        proof.append(item)
            if missing:
                out.append((f, axis, missing, proof))
    return out


def _axis_values(items: list[Any], axis: str) -> set[str]:
    """Distinct values an array's object items carry for one axis property."""
    return {
        str(item[axis])
        for item in items
        if isinstance(item, dict) and item.get(axis) not in (None, "")
    }


def _windows_holding_rows(window_texts: list[str], rows: list[Any]) -> list[int]:
    """Indices of windows whose text contains one of the rows.

    A row places in a window when at least :data:`_DECONFLICT_MIN_LEAF_MATCHES` of
    its leaves (each at least :data:`_DECONFLICT_LEAF_MIN_CHARS` chars, folded the
    same way grounding folds) appear in the window text.
    """
    leaf_sets = [
        [
            _ground_norm(s)
            for s in _string_leaves(row)
            if len(s.strip()) >= _DECONFLICT_LEAF_MIN_CHARS
        ]
        for row in rows
    ]
    indices: list[int] = []
    for i, text in enumerate(window_texts):
        norm = _ground_norm(text)
        for leaves in leaf_sets:
            if sum(1 for lf in leaves if lf and lf in norm) >= _DECONFLICT_MIN_LEAF_MATCHES:
                indices.append(i)
                break
    return indices


async def _continue_truncated_arrays(
    leaf: CapacityLeaf,
    provider: LLMProvider,
    state: PipelineState,
    extracted: dict[str, Any],
    truncated: set[str],
) -> None:
    """Recover the tail of an array the output token limit cut mid-list.

    One call cannot emit an array larger than its output budget, however the
    input is packed; the parser keeps the complete items, and the entries after
    the last one still sit in the document. That item is located in the segment
    stream (punctuation-folded substring) and windows from its segment onward
    are swept in document order - the same machinery that extends arrays past
    the excerpt, driven by output overflow instead of input overflow.
    """
    array_fields = [f for f in leaf.fields if f.path in truncated and _is_unbounded_list_leaf(f)]
    if not array_fields or not state.segments:
        return
    segments = sorted(state.segments, key=lambda s: s.start)
    norm_by_id = {s.segment_id: _ground_norm(s.text) for s in segments}
    anchor_start: int | None = None
    for f in array_fields:
        items = extracted.get(f.path)
        if isinstance(items, list) and items:
            seg = _last_item_segment(items, segments, norm_by_id)
            if seg is not None:
                anchor_start = seg.start if anchor_start is None else min(anchor_start, seg.start)
    if anchor_start is not None:
        # The anchor segment itself is included: the cut item may straddle it, and
        # the merge dedupe absorbs the items already captured before the cut.
        tail = [s for s in segments if s.start >= anchor_start]
    else:
        # Nothing salvaged or items unlocatable: extract afresh window by window
        # over the whole document, since output-sized calls sum where one cannot.
        tail = segments
    logger.info(
        "output-truncated array on leaf %d: %s - continuing over %d segment(s)%s",
        leaf.leaf_id,
        sorted(f.path for f in array_fields),
        len(tail),
        " from last salvaged item" if anchor_start is not None else " (full document)",
    )
    window_chars = _window_chars(leaf, state)
    preamble = _document_preamble(state, _preamble_cap(leaf, state, window_chars))
    windows = _pack_windows(tail, window_chars, {})
    await _sweep_array_windows(
        [text for text, _ in windows],
        array_fields,
        leaf,
        provider,
        state,
        extracted,
        preamble=preamble,
    )


def _last_item_segment(
    items: list[Any], segments: list[Any], norm_by_id: dict[int, str]
) -> Any | None:
    """The last document-order segment containing the array's last locatable item.

    An item grounds through one contiguous string: itself for a scalar, its
    longest string leaves for an object (the leaves of a row are scattered
    across the source line, so they are tried one at a time, never joined).
    """
    tried = 0
    for item in reversed(items):
        leaves = sorted(
            (s for s in _string_leaves(item) if len(s.strip()) >= _ANCHOR_ITEM_MIN_CHARS),
            key=len,
            reverse=True,
        )
        if not leaves:
            continue
        tried += 1
        if tried > _ANCHOR_ITEM_TRIES:
            return None
        for leaf_text in leaves[:_ANCHOR_LEAVES_PER_ITEM]:
            norm = _ground_norm(leaf_text)
            for seg in reversed(segments):
                if norm in norm_by_id[seg.segment_id]:
                    return seg
    return None


def _string_leaves(item: Any) -> list[str]:
    """Flatten an object item's string values for substring grounding."""
    if isinstance(item, str):
        return [item]
    if isinstance(item, dict):
        return [s for v in item.values() for s in _string_leaves(v)]
    if isinstance(item, list):
        return [s for v in item for s in _string_leaves(v)]
    return [str(item)] if item is not None else []


def _window_chars(leaf: CapacityLeaf, state: PipelineState) -> int:
    """Continuation window size, bounded by both the input and output budgets."""
    cpt = max(state.chars_per_token, 1.0)
    input_chars = int(max(0.0, state.C_usable - leaf.overhead) * cpt)
    safe_chars = safe_excerpt_chars(state.C_eff, leaf.overhead, leaf.safe_output, cpt)
    input_chars = min(input_chars, safe_chars)
    output_chars = int(leaf.safe_output * cpt * _OUTPUT_WINDOW_FACTOR)
    # Fall back to the input bound if the output bound rounds to zero.
    return max(1, min(input_chars, output_chars) or input_chars)


def _preamble_cap(leaf: CapacityLeaf, state: PipelineState, window_chars: int) -> int:
    """Characters the document-head preamble may add on top of a window.

    The window is output-bound (items are copied verbatim), so input capacity
    usually has slack beyond the window text; the preamble rides in that slack and
    never displaces content. Capped at the window fraction so a huge input budget
    still cannot bury the window under head text.
    """
    cpt = max(state.chars_per_token, 1.0)
    input_chars = int(max(0.0, state.C_usable - leaf.overhead) * cpt)
    safe_chars = safe_excerpt_chars(state.C_eff, leaf.overhead, leaf.safe_output, cpt)
    slack = max(0, min(input_chars, safe_chars) - window_chars)
    return min(int(window_chars * _PREAMBLE_WINDOW_FRACTION), slack)


def _document_preamble(state: PipelineState, cap: int) -> str:
    """Document-head context prepended to continuation windows.

    Values deep in a document are labelled by facts its opening states once (title
    block, covered period, declared units or scope); a window cut below them loses
    those facts. Leading whole segments are taken until the cap; a first segment
    larger than the cap is cut at it, since identity lines sit at its top.

    Args:
        state: Pipeline state carrying the document segments.
        cap: Maximum preamble characters (see :func:`_preamble_cap`).

    Returns:
        The head text, or ``""`` when there are no segments or no budget.
    """
    if cap <= 0 or not state.segments:
        return ""
    parts: list[str] = []
    size = 0
    for seg in sorted(state.segments, key=lambda s: s.start):
        if not parts and len(seg.text) > cap:
            return seg.text[:cap]
        if size + len(seg.text) > cap:
            break
        parts.append(seg.text)
        size += len(seg.text)
    return "\n\n".join(parts)


def _pack_windows(
    segments: list[Any], window_chars: int, score_by_id: dict[int, float]
) -> list[tuple[str, float]]:
    """Pack document-order segments into windows of at most *window_chars*.

    Consecutive windows overlap by one segment so an entry straddling the
    boundary is seen whole by the next window; the merge dedupe absorbs the
    repeats. Each window carries the sum of its segments' retrieval scores.
    """
    windows: list[tuple[str, float]] = []
    current: list[str] = []
    size = 0
    relevance = 0.0
    for seg in segments:
        if current and size + len(seg.text) > window_chars:
            windows.append(("\n\n".join(current), relevance))
            current, size, relevance = [current[-1]], len(current[-1]), 0.0
        current.append(seg.text)
        size += len(seg.text)
        relevance += score_by_id.get(seg.segment_id, 0.0)
    if current:
        windows.append(("\n\n".join(current), relevance))
    return windows


def _continuation_reason(field: Field) -> str:
    """Per-field steering for a continuation window, matched to the array's shape.

    A DIMENSION array (its items carry a categorical enum axis) is under-filled when
    only the aggregate row was found; this portion may report the value broken out by
    that axis, so the model is asked for one entry per distinct axis value shown here.
    Any other array (a bibliography-style list) is asked to copy each entry's full
    text verbatim; the field's own description then bounds what one entry IS, so a
    name list yields names rather than the blocks around them. Schema-driven, so
    neither framing is a hardcoded domain rule - the axis meaning comes from the
    schema's enum values and the entry meaning from the schema's description.
    """
    axes = dimension_axes(field)
    if axes:
        dims = " and ".join(f"{name} ([{', '.join(values)}])" for name, values in axes)
        return (
            "this is another portion of the same document; it may report this value "
            f"broken out by {dims}. Output one entry for EACH distinct value of that "
            "axis shown in this portion, using the exact keys and shape - one entry "
            "per row, never merged. Output [] only when this portion reports no such "
            "entry"
        )
    items = field.constraints.get("items")
    desc = items.get("description") if isinstance(items, dict) else None
    desc = desc or field.schema_node.get("description")
    entry_is = f" One entry here means: {desc}." if isinstance(desc, str) and desc else ""
    return (
        "this is another portion of the same document and may hold a slice of "
        "this array's entries; copy the complete text of every entry that "
        "appears here, however many there are, ONE item per entry - never merge "
        "consecutive entries into one item. A reference marker, a bare number, "
        "or a mention of an entry is NOT an entry." + entry_is + " Output each "
        "item exactly as the document writes it, complete - keep abbreviations, "
        "punctuation, and any suffix, branch, or parenthetical that is part of "
        "the entry itself - but end the item where the entry ends: do not append "
        "clauses that merely describe or classify it, and do not include the "
        "next entry. One line per item. Output [] only when no entries appear "
        "in this portion"
    )


async def _sweep_array_windows(
    window_texts: list[str],
    array_fields: list[Field],
    leaf: CapacityLeaf,
    provider: LLMProvider,
    state: PipelineState,
    extracted: dict[str, Any],
    visit_order: list[int] | None = None,
    preamble: str = "",
    reasons: dict[str, str] | None = None,
    cap_bonus: int = 0,
) -> None:
    """Ask each window only for the array fields and merge new items with dedupe.

    Windows are visited in *visit_order* (default: given order). A window that
    yields items promotes its unvisited DOCUMENT-order neighbours to the front:
    items cluster, so a productive window's neighbour is the best next probe -
    the whole cluster is walked before the ordering moves elsewhere. Stops after
    :data:`_CONTINUATION_STOP_AFTER_EMPTY` consecutive windows yield almost
    nothing, never exceeding the document-sized per-document window budget.

    *preamble* (the document head) is prepended to every window except one that
    already starts with it, so each call keeps the document's global identity.
    """
    reasons = reasons or {f.path: _continuation_reason(f) for f in array_fields}
    from nfield.schema._types import CapacityLeaf

    # Document-sized shared budget: a schema with many empty arrays would otherwise
    # sweep once per leaf and multiply the call count without bound. A rescue pass
    # (deterministic trigger, few windows) is granted its own headroom over whatever
    # is already spent, so earlier passes exhausting the budget cannot silence it.
    per_doc_cap = _max_continuation_windows_per_doc(leaf, state)
    if cap_bonus:
        per_doc_cap = max(per_doc_cap, state.continuation_windows_used + cap_bonus)

    async def probe_text(window_text: str, idx: int) -> tuple[int, int] | None:
        """One window call; merge new items; return (raw count, new-item yield)."""
        if state.continuation_windows_used >= per_doc_cap:
            return None
        state.continuation_windows_used += 1
        # Reserve output sized to the window, not the array's ceiling: items are copied
        # verbatim, and booking the ceiling eats the provider's per-minute token budget.
        cpt = max(state.chars_per_token, 1.0)
        window_output = min(
            leaf.safe_output,
            max(1, math.ceil(len(window_text) * _WINDOW_OUTPUT_HEADROOM / cpt)),
        )
        excerpt = window_text
        if preamble and not window_text.startswith(preamble):
            excerpt = f"{preamble}\n\n{window_text}"
        window_leaf = CapacityLeaf(
            fields=array_fields,
            groups=[],
            document_excerpt=excerpt,
            overhead=leaf.overhead,
            safe_output=window_output,
            leaf_id=leaf.leaf_id,
        )
        try:
            raw = await _call_provider(window_leaf, provider, state, field_reasons=reasons)
        except Exception as exc:
            logger.warning("Array continuation window failed on leaf %d: %s", leaf.leaf_id, exc)
            return None
        state.record_calls("array_continuation")
        parsed = parse_sfep(raw, array_fields)
        parsed = await _retry_ordinal_runs(parsed, window_leaf, provider, state, reasons)
        raw_count = 0
        window_yield = 0
        for f in array_fields:
            more = parsed.get(f.path)
            if not isinstance(more, list) or not more or _is_ordinal_run(more):
                continue
            raw_count += len(more)
            base = extracted.get(f.path)
            merged = list(base) if isinstance(base, list) else []
            window_yield += _merge_window_items(merged, more)
            extracted[f.path] = merged
        logger.info(
            "continuation window %d/%d (doc idx %d) on leaf %d (%d chars): +%d item(s), %d raw%s for %s",
            state.continuation_windows_used,
            per_doc_cap,
            idx,
            leaf.leaf_id,
            len(window_text),
            window_yield,
            raw_count,
            " TRUNCATED" if truncated_json_arrays(raw, array_fields) else "",
            [f.path for f in array_fields],
        )
        return raw_count, window_yield

    raw_counts: dict[int, int] = {}

    async def probe(idx: int) -> int | None:
        """Probe window *idx*; record its raw density; return the new-item yield."""
        result = await probe_text(window_texts[idx], idx)
        if result is None:
            return None
        raw, merged = result
        raw_counts[idx] = max(raw_counts.get(idx, 0), raw)
        return merged

    consecutive_empty = 0
    order = list(visit_order) if visit_order is not None else list(range(len(window_texts)))
    promoted: list[int] = []
    visited: set[int] = set()
    yields: dict[int, int] = {}
    pos = 0
    calls = 0
    while (promoted or pos < len(order)) and calls < per_doc_cap:
        if promoted:
            idx = promoted.pop(0)
        else:
            idx = order[pos]
            pos += 1
        if idx in visited:
            continue
        visited.add(idx)
        calls += 1
        window_yield = await probe(idx)
        if window_yield is None:
            continue
        yields[idx] = window_yield
        if window_yield >= _CONTINUATION_MIN_YIELD:
            consecutive_empty = 0
            # Items cluster: a productive window's document neighbours are the best
            # next probes, so the whole cluster is walked before moving elsewhere.
            promoted.extend(
                nb
                for nb in (idx - 1, idx + 1)
                if 0 <= nb < len(window_texts) and nb not in visited
            )
        else:
            consecutive_empty += 1
        aboard = sum(len(v) for f in array_fields if isinstance(v := extracted.get(f.path), list))
        scarce = aboard < _CONTINUATION_SCARCE_ITEMS
        patience = _CONTINUATION_STOP_WHILE_EMPTY if scarce else _CONTINUATION_STOP_AFTER_EMPTY
        if consecutive_empty >= patience:
            break

    # Cover enumerations and signature pages evade relevance and patience, so the
    # outermost unvisited windows are always verified; a yielding end walks its neighbour.
    unvisited = [i for i in range(len(window_texts)) if i not in visited]
    for idx in dict.fromkeys(unvisited[-1:] + unvisited[:1]):
        visited.add(idx)
        if await probe(idx):
            for nb in (idx - 1, idx + 1):
                if 0 <= nb < len(window_texts) and nb not in visited:
                    visited.add(nb)
                    await probe(nb)

    async def split_reprobe(idx: int) -> None:
        """Re-ask window *idx* in two halves; each half is short enough to enumerate."""
        text = window_texts[idx]
        cut = text.rfind("\n\n", 0, len(text) // 2)
        if cut <= 0:
            cut = len(text) // 2
        recovered = 0
        for part in (text[:cut], text[cut:]):
            result = await probe_text(part, idx)
            if result is not None:
                recovered += result[1]
        if recovered:
            logger.info("low-yield window %d split resample: +%d item(s)", idx, recovered)

    # A window with far fewer RAW items than a probed neighbour is under-enumerated,
    # not sparse (raw counts see density a deduped yield hides); split-reprobe it.
    for idx in sorted(raw_counts):
        neighbours = [raw_counts[nb] for nb in (idx - 1, idx + 1) if nb in raw_counts]
        if not neighbours:
            continue
        floor = min(neighbours)
        if floor < 3 * _CONTINUATION_MIN_YIELD or raw_counts[idx] * 2 >= floor:
            continue
        await split_reprobe(idx)

    # A <2-window sweep has no neighbour to expose under-emission; a still-empty
    # array gets one split re-probe of the probed window.
    if len(raw_counts) < 2 and any(not extracted.get(f.path) for f in array_fields):
        for idx in sorted(raw_counts):
            await split_reprobe(idx)

    # Windows are visited in relevance order, so the merged list carries visit
    # order; restore document order - the order the entries are read in.
    for f in array_fields:
        items = extracted.get(f.path)
        if isinstance(items, list) and len(items) > 1:
            extracted[f.path] = _document_order(items, state)


def _document_order(items: list[Any], state: PipelineState) -> list[Any]:
    """Sort items by the position of their text in the document.

    An item locates by its longest string leaf (punctuation-folded substring);
    items that cannot be located keep their relative order after the located
    ones.
    """
    doc_norm = _ground_norm(
        "\n".join(s.text for s in sorted(state.segments, key=lambda s: s.start))
    )
    keyed: list[tuple[int, int, Any]] = []
    for i, item in enumerate(items):
        leaves = sorted(_string_leaves(item), key=len, reverse=True)
        at = doc_norm.find(_ground_norm(leaves[0])) if leaves and leaves[0].strip() else -1
        keyed.append((at if at >= 0 else len(doc_norm) + i, i, item))
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [item for _, _, item in keyed]


def _item_key(item: Any) -> str:
    """Stable equality key for an array item (dicts compare by sorted JSON)."""
    if isinstance(item, dict):
        return json.dumps(item, sort_keys=True, ensure_ascii=False)
    return str(item)


def _is_ordinal_run(items: list[Any]) -> bool:
    """True when the list is mostly consecutive integers - numbering, not data."""
    nums = []
    for x in items:
        if isinstance(x, bool):
            return False
        if isinstance(x, int):
            nums.append(x)
        elif isinstance(x, str) and x.strip().isdigit():
            nums.append(int(x.strip()))
    if len(nums) < 3 or len(nums) < _ORDINAL_RUN_MIN_FRACTION * len(items):
        return False
    ascending = sum(1 for a, b in itertools.pairwise(nums) if b == a + 1)
    return ascending >= _ORDINAL_RUN_MIN_FRACTION * (len(nums) - 1)


async def _retry_ordinal_runs(
    parsed: dict[str, Any],
    window_leaf: CapacityLeaf,
    provider: LLMProvider,
    state: PipelineState,
    reasons: dict[str, str],
) -> dict[str, Any]:
    """Retry a window once when a field came back as its entry numbers.

    A consecutive-integer list is the document's numbering; one corrective call
    names the mistake. The retry is adopted only for fields that stop being runs.
    """
    run_paths = {
        f.path
        for f in window_leaf.fields
        if isinstance(parsed.get(f.path), list) and _is_ordinal_run(parsed[f.path])
    }
    if not run_paths:
        return parsed
    sharpened = dict(reasons)
    for path in run_paths:
        sharpened[path] = (
            "the previous answer gave the entries' NUMBERS; output each entry's "
            "complete text exactly as written, never its number or label"
        )
    try:
        raw = await _call_provider(window_leaf, provider, state, field_reasons=sharpened)
    except Exception as exc:
        logger.warning("Ordinal-run retry failed on leaf %d: %s", window_leaf.leaf_id, exc)
        return parsed
    state.record_calls("array_continuation")
    retried = parse_sfep(raw, window_leaf.fields)
    for path in run_paths:
        value = retried.get(path)
        if isinstance(value, list) and value and not _is_ordinal_run(value):
            parsed[path] = value
    return parsed


def _merge_window_items(merged: list[Any], more: list[Any]) -> int:
    """Merge a window's items into *merged*; return how many were genuinely new.

    Overlapping windows re-emit the same entry in fuller or shorter form (a title
    vs the whole reference line), so string items dedupe by normalized containment
    and the LONGEST copy wins. An object row re-emitted with a field missing or added
    dedupes the same way: when one row's set of text values is a subset of another's
    they are the same entry, and the fuller row is kept. In a long-text list, an item
    that is only a bare number is a marker copied from the document, not an entry, and
    is skipped. String items carry the source's layout line breaks when copied
    verbatim; internal whitespace runs collapse to one space so a value is text.
    """
    more = [_WS.sub(" ", x).strip() if isinstance(x, str) else x for x in more]
    long_text = sum(
        1 for x in merged + more if isinstance(x, str) and len(x) >= _LONG_TEXT_ITEM_CHARS
    )
    drop_bare = long_text >= max(3, (len(merged) + len(more)) // 2)
    keys = {_item_key(x) for x in merged}
    norms: list[str | None] = [_ground_norm(x) if isinstance(x, str) else None for x in merged]
    leaves: list[frozenset[str] | None] = [
        _object_leaves(x) if isinstance(x, dict) else None for x in merged
    ]
    added = 0
    for item in more:
        key = _item_key(item)
        if key in keys:
            continue
        bare_number = (isinstance(item, int) and not isinstance(item, bool)) or (
            isinstance(item, str) and item.strip().isdigit()
        )
        if drop_bare and bare_number:
            continue
        item_leaves: frozenset[str] | None = None
        if isinstance(item, str):
            norm = _ground_norm(item)
            dup = False
            for i, existing in enumerate(norms):
                if existing is None or not existing or not norm:
                    continue
                if norm in existing:
                    dup = True  # a shorter copy of an entry already aboard
                    break
                if existing in norm:
                    merged[i] = item  # keep the longer copy of the same entry
                    norms[i] = norm
                    dup = True
                    break
            if dup:
                keys.add(key)
                continue
        elif isinstance(item, dict):
            item_leaves = _object_leaves(item)
            if item_leaves:
                dup = False
                for i, existing_leaves in enumerate(leaves):
                    if not existing_leaves:
                        continue
                    if item_leaves <= existing_leaves:
                        dup = True  # same entry; the stored copy is as full or fuller
                        break
                    if existing_leaves < item_leaves:
                        merged[i] = item  # keep the fuller copy of the same entry
                        leaves[i] = item_leaves
                        dup = True
                        break
                if dup:
                    keys.add(key)
                    continue
        keys.add(key)
        merged.append(item)
        norms.append(_ground_norm(item) if isinstance(item, str) else None)
        leaves.append(item_leaves)
        added += 1
    return added


def _object_leaves(obj: Any) -> frozenset[str]:
    """Normalized non-empty string values of an object, for near-duplicate detection."""
    return frozenset(n for s in _string_leaves(obj) if (n := _ground_norm(s)))


def _is_unbounded_list_leaf(f: Field) -> bool:
    """True for an array list-leaf (item schema present) without a maxItems bound."""
    return (
        f.type == "array"
        and isinstance(f.constraints.get("items"), dict)
        and not isinstance(f.constraints.get("maxItems"), int)
    )


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


async def _call_provider(
    leaf: CapacityLeaf,
    provider: LLMProvider,
    state: PipelineState,
    *,
    field_reasons: dict[str, str] | None = None,
) -> str:
    """Build prompt and call provider for a single leaf.

    Args:
        leaf: Leaf to extract.
        provider: LLM provider.
        state: Pipeline state (supplies caller system/user prompt context).
        field_reasons: Per-field steering for this call only. Passed explicitly by
            the continuation passes so concurrent leaves never share the mutable
            ``state.field_reasons``; falls back to it when not given.

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
        field_reasons=field_reasons or state.field_reasons or None,
    )
    raw = await provider.complete(messages, max_tokens=leaf.safe_output)
    _calibrate_chars_per_token(state, messages, provider)
    return raw


def _calibrate_chars_per_token(
    state: PipelineState, messages: list[dict[str, str]], provider: LLMProvider
) -> None:
    """Tighten the chars-per-token estimate from the model's real token count.

    The API returns the exact input token count for the call; dividing the prompt's
    character count by it gives this model's true tokenizer density. Tracking the
    densest value seen keeps later excerpts and windows within the real window
    without a per-model constant. A denser reading only shrinks the estimate (never
    loosens it), so calibration can never enlarge a call into an overflow.
    """
    prompt_tokens = getattr(provider, "last_prompt_tokens", None)
    if not prompt_tokens or prompt_tokens <= 0:
        return
    prompt_chars = sum(len(m.get("content", "")) for m in messages)
    observed = prompt_chars / prompt_tokens
    if observed < _MIN_CALIBRATED_CPT:
        return
    state.chars_per_token = min(state.chars_per_token, observed)


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
    *,
    split_depth: int = 0,
) -> None:
    """Recover an oversized leaf that overflowed the context window.

    A context overflow is input-driven (the excerpt tokenized to more than the
    estimate), so the first recovery is to halve the EXCERPT and retry the same
    fields: this keeps the field set whole, so an unbounded array keeps its
    continuation sweep over the rest of the document. Only when the excerpt is
    already minimal does the field set split, as a last resort. Each retry
    re-enters :func:`_extract_leaf` and may recover again up to
    :data:`_MAX_SPLIT_DEPTH`.

    Args:
        leaf: Oversized leaf to recover.
        provider: LLM provider.
        state: Pipeline state.
        split_depth: Recursion depth of the leaf being recovered.
    """
    from nfield.schema._types import CapacityLeaf

    assert state.blackboard is not None
    half_excerpt = leaf.document_excerpt[: max(1, len(leaf.document_excerpt) // 2)]
    covered = _excerpt_prefix_ids(leaf, len(half_excerpt), state)

    # Halve the excerpt while it exceeds the reserved output (cutting input helps);
    # once smaller, the overflow is output/field-driven, so split the field set instead.
    excerpt_tokens = len(leaf.document_excerpt) / max(state.chars_per_token, 1.0)
    if excerpt_tokens > leaf.safe_output and split_depth < _MAX_SPLIT_DEPTH:
        smaller = CapacityLeaf(
            fields=leaf.fields,
            groups=leaf.groups,
            document_excerpt=half_excerpt,
            overhead=leaf.overhead,
            safe_output=leaf.safe_output,
            leaf_id=leaf.leaf_id,
            excerpt_segment_ids=covered,
        )
        await _extract_leaf(smaller, provider, state, split_depth=split_depth + 1)
        return

    # Excerpt is already minimal - the field set itself must be too large. Split it.
    mid = max(1, len(leaf.fields) // 2)
    for chunk_fields in (leaf.fields[:mid], leaf.fields[mid:]):
        if not chunk_fields:
            continue
        split_leaf = CapacityLeaf(
            fields=chunk_fields,
            groups=leaf.groups,
            document_excerpt=half_excerpt,
            overhead=leaf.overhead,
            safe_output=leaf.safe_output,
            leaf_id=leaf.leaf_id,
            excerpt_segment_ids=covered,
        )
        await _extract_leaf(split_leaf, provider, state, split_depth=split_depth + 1)


def _excerpt_prefix_ids(leaf: CapacityLeaf, prefix_len: int, state: PipelineState) -> set[int]:
    """Segment ids whose text fits inside the excerpt's first *prefix_len* chars.

    The excerpt concatenates its segments in document order, so the prefix covers
    a document-ordered prefix of them. Under-inclusion is safe: a segment dropped
    here is re-swept by the array window extension, and dedupe absorbs overlap.
    """
    by_id = {s.segment_id: s for s in state.segments}
    used = 0
    ids: set[int] = set()
    for sid in sorted(leaf.excerpt_segment_ids, key=lambda i: by_id[i].start if i in by_id else 0):
        seg = by_id.get(sid)
        if seg is None:
            continue
        if used + len(seg.text) > prefix_len:
            break
        ids.add(sid)
        used += len(seg.text)
    return ids


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
            # One recovery call cannot out-collect a windowed sweep, so a shorter
            # re-extraction is a partial redo; keep the fuller quality-failed original.
            stashed = state.quality_failed_values.get(path)
            if isinstance(value, list) and isinstance(stashed, list) and len(value) < len(stashed):
                value = stashed
            # Canonicalize a formatted value to its schema type before storing, so a
            # number field holds a number (schema-valid output) and validation does not
            # reject it on format alone. Lossless-or-decline; skipped in strict mode.
            field = state.field_by_path.get(path)
            if field is not None and not state.strict_validation:
                value = normalize_value(value, field)
            state.blackboard.write(path, value)
