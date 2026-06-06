"""Surgical Field Retry (SFR) — re-extract only failed fields.

SFR is Innovation INN_003 of FormatShield. Instead of re-running the full
extraction pass when some fields fail validation, SFR constructs a targeted
retry prompt containing *only* the failed fields along with their specific
validation error messages. This reduces retry cost by (1 - R/K), where R is
the number of fields that need retry and K is the total field count.

Two rounds suffice in practice: the probability a field still fails after two
independent retries is low (~2.4% under the project's measured per-round rates).

MVP failure causes (4)
----------------------
* ``FORMAT``                   — SFEP parse failed; malformed key=value line.
* ``TYPE_CONSTRAINT``          — Field extracted but type/constraint invalid.
* ``FIELD_MISSING``            — Field absent from LLM output (EMPTY state).
* ``DEPENDENCY_VALUE_CHANGED`` — A dependency's value changed during retry,
                                  invalidating this field's extracted value.

Post-MVP stubs (not implemented)
---------------------------------
* ``LOW_GROUNDING_EVIDENCE_PRESENT`` — GSV grounding score below threshold.
* ``LOW_GROUNDING_NO_EVIDENCE``      — No supporting evidence found in document.
* PFTEN (pool-first narrowed excerpt) — targeted context narrowing per field.
* GSGRF (targeted retrieval)         — per-field BM25 re-query.
* CADTR (cascade dep invalidation)   — dependency-aware invalidation tree.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Any

from formatshield.extraction._prompt import build_retry_system_message
from formatshield.extraction._sfep import parse_sfep

if TYPE_CHECKING:
    from formatshield.assembly._blackboard import Blackboard
    from formatshield.config import ExtractionConfig
    from formatshield.providers._protocol import LLMProvider
    from formatshield.schema._types import CapacityLeaf, Field

__all__ = [
    "FailureCause",
    "build_retry_prompt",
    "cascade_invalidate",
    "classify_failure",
    "handle_missing_fields",
    "orchestrate_retry",
    "split_retry_batches",
    "surgical_field_retry",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RETRY_ROUNDS: int = 2
_FORMAT_ERROR_KEYWORDS: frozenset[str] = frozenset(
    {"parse", "format", "sfep", "malformed", "separator", "key=value"}
)
# Per-line allowance for a retry output line's " = " separator + newline, added
# on top of the field's value (tau) and echoed path when packing retry batches.
_RETRY_LINE_OVERHEAD_TOKENS: int = 8


# ---------------------------------------------------------------------------
# FailureCause enum
# ---------------------------------------------------------------------------


class FailureCause(Enum):
    """Root cause classification for a failed field extraction.

    Attributes:
        FORMAT: SFEP parse error — the line was not a valid ``path = value`` pair.
        TYPE_CONSTRAINT: The extracted value failed type or constraint validation.
        FIELD_MISSING: The field was absent from the LLM output (never extracted).
        DEPENDENCY_VALUE_CHANGED: A dependency field was updated in a later
            extraction round, invalidating this field's value.

    Example:
        >>> FailureCause.FORMAT.value
        'format'
    """

    FORMAT = "format"
    TYPE_CONSTRAINT = "type_constraint"
    FIELD_MISSING = "field_missing"
    DEPENDENCY_VALUE_CHANGED = "dependency_value_changed"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_failure(
    field: Field,
    value: Any,
    error: str,
) -> FailureCause:
    """Classify the root cause of a field extraction failure.

    Uses heuristics on the error message to distinguish format errors from
    constraint violations. Both ``FIELD_MISSING`` and
    ``DEPENDENCY_VALUE_CHANGED`` require external context and must be set
    by the caller.

    Args:
        field: The field that failed extraction.
        value: The extracted value (may be ``None`` for missing fields).
        error: Validation error message from :func:`~formatshield.validation._type_check.validate_field`.

    Returns:
        The most likely :class:`FailureCause` for this failure.

    Example:
        >>> from formatshield.schema._types import Field
        >>> f = Field("age", "integer", {}, "", {})
        >>> classify_failure(f, None, "field_missing")
        <FailureCause.FIELD_MISSING: 'field_missing'>
        >>> classify_failure(f, "thirty", "expected integer")
        <FailureCause.TYPE_CONSTRAINT: 'type_constraint'>
    """
    error_lower = error.lower()

    if "field_missing" in error_lower or ("missing" in error_lower and value is None):
        return FailureCause.FIELD_MISSING

    if "dependency" in error_lower or "dep_changed" in error_lower:
        return FailureCause.DEPENDENCY_VALUE_CHANGED

    # Check for SFEP format errors
    if any(kw in error_lower for kw in _FORMAT_ERROR_KEYWORDS):
        return FailureCause.FORMAT

    # Default: treat as type/constraint violation
    return FailureCause.TYPE_CONSTRAINT


async def orchestrate_retry(
    failed_fields: list[Field],
    errors: dict[str, str],
    provider: LLMProvider,
    leaf: CapacityLeaf,
    *,
    dep_dag: dict[str, set[str]],
    config: ExtractionConfig,
    call_counter: list[int] | None = None,
    system_prompt: str = "",
    user_prompt: str = "",
    knowledge_fallback: bool = False,
    retry_excerpt: str | None = None,
) -> dict[str, Any]:
    """Orchestrate up to 2 rounds of surgical field retry.

    Runs failed fields through targeted retry calls with specific error
    context. Max rounds is bounded by ``config.max_retry_rounds`` (default 2).
    Returns a dict of recovered ``{path: value}`` pairs.

    In MVP, all four failure causes (FORMAT, TYPE_CONSTRAINT, FIELD_MISSING,
    DEPENDENCY_VALUE_CHANGED) route to the same handler: same-excerpt retry
    with the error message as context. Post-MVP will add cause-specific
    handlers (PFTEN, GSGRF, CADTR).

    Args:
        failed_fields: Fields that failed validation in Stage 4.
        errors: Mapping of ``field.path -> error_message`` for each failure.
        provider: LLM provider for retry API calls.
        leaf: The capacity leaf whose excerpt is used for retry.
        dep_dag: Dependency graph used to batch fields with shared dependencies.
        config: Extraction configuration controlling max retry rounds.
        call_counter: Optional single-element list; when given, element 0 is
            incremented once per provider call made, so callers can fold retry
            cost into the run's total API-call count.

    Returns:
        Dict of ``{path: value}`` for fields that recovered in retry rounds.
        Fields that remain failing after all rounds are absent from the dict.

    Example:
        >>> # (see tests/unit/validation/test_retry.py for async usage)
    """
    if not failed_fields:
        return {}

    max_rounds = config.max_retry_rounds
    recovered: dict[str, Any] = {}
    still_failing = list(failed_fields)
    current_errors = dict(errors)

    for round_idx in range(max_rounds):
        if not still_failing:
            break

        logger.debug(
            "SFR round %d/%d: retrying %d field(s) in leaf %d",
            round_idx + 1,
            max_rounds,
            len(still_failing),
            leaf.leaf_id,
        )

        batches = split_retry_batches(
            still_failing, dep_dag, max_output_tokens=leaf.safe_output or None
        )
        round_recovered: dict[str, Any] = {}

        for batch in batches:
            batch_result = await surgical_field_retry(
                batch,
                {p: current_errors.get(p, "validation failed") for p in (f.path for f in batch)},
                provider,
                leaf,
                call_counter=call_counter,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                knowledge_fallback=knowledge_fallback,
                retry_excerpt=retry_excerpt,
            )
            round_recovered.update(batch_result)

        recovered.update(round_recovered)

        # Remove recovered fields from next round
        recovered_paths = set(round_recovered)
        still_failing = [f for f in still_failing if f.path not in recovered_paths]
        current_errors = {p: e for p, e in current_errors.items() if p not in recovered_paths}

    if still_failing:
        logger.debug(
            "SFR: %d field(s) still failing after %d round(s): %s",
            len(still_failing),
            max_rounds,
            [f.path for f in still_failing],
        )

    return recovered


async def surgical_field_retry(
    fields: list[Field],
    errors: dict[str, str],
    provider: LLMProvider,
    leaf: CapacityLeaf,
    *,
    call_counter: list[int] | None = None,
    system_prompt: str = "",
    user_prompt: str = "",
    knowledge_fallback: bool = False,
    retry_excerpt: str | None = None,
) -> dict[str, Any]:
    """Execute one surgical retry call for a batch of failed fields.

    Builds a targeted retry prompt with per-field error context and sends
    a single API call to recover the failed fields.

    Args:
        fields: Failed fields to retry (one batch from split_retry_batches).
        errors: Per-field error messages.
        provider: LLM provider for the retry call.
        leaf: Capacity leaf providing the document excerpt.
        call_counter: Optional single-element list; element 0 is incremented by
            one for the API call this function makes (for cost accounting).
        retry_excerpt: Fresh, field-targeted excerpt from re-retrieval (GSGRF).
            When given it replaces the leaf's original excerpt, so a field whose
            evidence was trimmed away gets a different, relevant context. Falls
            back to ``leaf.document_excerpt`` when ``None``.

    Returns:
        Dict of ``{path: value}`` for fields successfully re-extracted.
    """
    messages = build_retry_prompt(
        fields,
        errors,
        retry_excerpt or leaf.document_excerpt,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        knowledge_fallback=knowledge_fallback,
    )

    # Estimate output tokens: sum of tau values + safety margin
    total_tau = sum(f.tau for f in fields)
    max_tokens = max(leaf.safe_output, int(total_tau * 2) + 50)

    if call_counter is not None:
        call_counter[0] += 1  # one provider call is about to be made

    try:
        raw_output = await provider.complete(messages, max_tokens=max_tokens)
    except Exception as exc:
        logger.warning(
            "SFR API call failed for leaf %d: %s",
            leaf.leaf_id,
            exc,
        )
        return {}

    result = parse_sfep(raw_output, fields)
    return result


def build_retry_prompt(
    fields: list[Field],
    errors: dict[str, str],
    document_excerpt: str,
    *,
    system_prompt: str = "",
    user_prompt: str = "",
    knowledge_fallback: bool = False,
) -> list[dict[str, str]]:
    """Build the messages list for a surgical retry call.

    Args:
        fields: Failed fields to retry.
        errors: Per-field error messages for targeted correction.
        document_excerpt: Document text from the original leaf.
        system_prompt: Optional caller system context (prepended).
        user_prompt: Optional caller task context (prepended).
        knowledge_fallback: Allow the model to fall back to its own knowledge for
            fields the document does not state. Default ``False``.

    Returns:
        Messages list for ``provider.complete()``.
    """
    return build_retry_system_message(
        fields,
        errors,
        document_excerpt,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        knowledge_fallback=knowledge_fallback,
    )


def _retry_output_tokens(field: Field) -> int:
    """Rough output-token cost of retrying one field's ``path = value`` line.

    Mirrors the packer's output model (echoed path + value + line overhead) so
    retry batching uses the same notion of cost; kept self-contained here to
    avoid a dependency on the packing module.

    Args:
        field: The field being retried.

    Returns:
        Estimated output tokens for this field's SFEP line.
    """
    return math.ceil(field.tau) + max(1, len(field.path) // 4) + _RETRY_LINE_OVERHEAD_TOKENS


def split_retry_batches(
    failed_fields: list[Field],
    dep_dag: dict[str, set[str]],
    *,
    max_output_tokens: int | None = None,
) -> list[list[Field]]:
    """Group failed fields into retry batches, preserving dependency closures.

    Two steps:

    1. **Closures (indivisible units).** Union-find over dependency edges among
       the failed fields keeps a field and its failed dependencies together, so
       dependency-aware re-extraction stays correct.
    2. **Capacity packing.** When ``max_output_tokens`` is given, those closures
       are greedily packed into the fewest batches whose combined output fits the
       budget — so many independent fields share one retry call instead of one
       call each. Without a budget, each closure is its own batch (legacy
       behaviour). A single closure that alone exceeds the budget still forms its
       own batch (a closure is never split).

    Args:
        failed_fields: Fields to batch.
        dep_dag: Full dependency graph ``{path: set_of_paths_it_depends_on}``.
        max_output_tokens: Per-call output budget for packing closures. ``None``
            disables packing (one batch per closure).

    Returns:
        List of batches; each batch is retried in a single API call.

    Example:
        >>> from formatshield.schema._types import Field
        >>> f1 = Field("a", "string", {}, "", {}, tau=2.0)
        >>> f2 = Field("b", "string", {}, "", {}, tau=2.0)
        >>> # No budget → one batch per independent field (legacy).
        >>> len(split_retry_batches([f1, f2], {}))
        2
        >>> # With a budget the two independent fields share one retry call.
        >>> len(split_retry_batches([f1, f2], {}, max_output_tokens=1000))
        1
    """
    if not failed_fields:
        return []

    failed_paths = {f.path for f in failed_fields}
    path_to_field: dict[str, Field] = {f.path: f for f in failed_fields}

    # Build adjacency among failed fields only
    adjacency: dict[str, set[str]] = {p: set() for p in failed_paths}
    for path in failed_paths:
        deps = dep_dag.get(path, set())
        for dep in deps:
            if dep in failed_paths:
                adjacency[path].add(dep)
                adjacency[dep].add(path)

    # Union-find: group connected components
    parent: dict[str, str] = {p: p for p in failed_paths}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def _union(x: str, y: str) -> None:
        parent[_find(x)] = _find(y)

    for path, neighbors in adjacency.items():
        for neighbor in neighbors:
            _union(path, neighbor)

    # Group by root → dependency closures (indivisible units)
    closures_by_root: dict[str, list[Field]] = {}
    for path in failed_paths:
        closures_by_root.setdefault(_find(path), []).append(path_to_field[path])
    closures = list(closures_by_root.values())

    if max_output_tokens is None or max_output_tokens <= 0:
        return closures

    # Capacity-pack closures into the fewest budget-bounded batches (heaviest
    # first), so independent fields are retried together rather than one-per-call.
    closures.sort(key=lambda c: sum(_retry_output_tokens(f) for f in c), reverse=True)
    batches: list[list[Field]] = []
    current: list[Field] = []
    current_cost = 0
    for closure in closures:
        cost = sum(_retry_output_tokens(f) for f in closure)
        if current and current_cost + cost > max_output_tokens:
            batches.append(current)
            current, current_cost = [], 0
        current.extend(closure)
        current_cost += cost
    if current:
        batches.append(current)
    return batches


def handle_missing_fields(
    missing_paths: list[str],
    leaf: CapacityLeaf,
    all_fields: list[Field],
) -> dict[str, Any]:
    """Attempt tree backtrack for fields that remain EMPTY after extraction.

    For each missing field, walks up the dot-notation path tree to check
    if a parent field exists. If the parent is also missing/None, marks
    the field as ``None`` (the parent context does not exist in the document).

    Args:
        missing_paths: Dot-notation paths of fields still EMPTY after SFR.
        leaf: The leaf containing the fields (provides field list context).
        all_fields: Complete list of all fields including those not in this leaf
            (used for parent-path existence checks).

    Returns:
        Dict of ``{path: None}`` for fields confirmed absent via tree backtrack.

    Example:
        >>> from formatshield.schema._types import CapacityLeaf
        >>> leaf = CapacityLeaf(fields=[], document_excerpt="", safe_output=0, leaf_id=0)
        >>> handle_missing_fields([], leaf, [])
        {}
    """
    if not missing_paths:
        return {}

    leaf_field_paths = {f.path for f in leaf.fields}
    result: dict[str, Any] = {}

    for path in missing_paths:
        # Walk up the dot-notation tree
        parts = path.split(".")
        if len(parts) <= 1:
            # Top-level field: no parent to check — mark as None
            result[path] = None
            continue

        # Check if any ancestor in the leaf is also missing
        for depth in range(len(parts) - 1, 0, -1):
            parent_path = ".".join(parts[:depth])
            if parent_path in leaf_field_paths:
                # Parent exists but is not in missing paths — child should have value
                break
        else:
            # No parent in this leaf — field truly absent
            result[path] = None

    return result


def cascade_invalidate(
    blackboard: Blackboard,
    dep_dag: dict[str, set[str]],
    changed_paths: set[str],
) -> list[str]:
    """Invalidate fields downstream of a changed dependency value (CADTR).

    When a retry round changes a value that other fields depend on, those
    dependents may be stale. This walks the *reverse* dependency graph from
    each changed path and marks every currently-FILLED dependent
    ``NEEDS_REVALIDATION`` (which removes it from the assembled output and
    surfaces it in metadata). Cascades transitively: a newly invalidated field
    invalidates its own dependents in turn.

    Args:
        blackboard: The run's blackboard (mutated in place).
        dep_dag: Field dependency graph ``{path: set_of_paths_it_depends_on}``.
        changed_paths: Paths whose values changed during retry.

    Returns:
        Sorted list of paths flagged ``NEEDS_REVALIDATION`` by this call.

    Example:
        >>> from formatshield.assembly._blackboard import Blackboard
        >>> bb = Blackboard(["total", "tax"])
        >>> bb.write("total", 100); bb.write("tax", 9)
        >>> cascade_invalidate(bb, {"tax": {"total"}}, {"total"})
        ['tax']
    """
    from formatshield.assembly._blackboard import FieldState

    # Reverse edges: dependents_of[d] = every field that depends on d.
    dependents_of: dict[str, set[str]] = {}
    for path, deps in dep_dag.items():
        for dep in deps:
            dependents_of.setdefault(dep, set()).add(path)

    invalidated: list[str] = []
    seen: set[str] = set(changed_paths)
    queue: deque[str] = deque(changed_paths)
    while queue:
        changed = queue.popleft()
        for dependent in dependents_of.get(changed, set()):
            if dependent in seen:
                continue
            seen.add(dependent)
            if blackboard.get_state(dependent) == FieldState.FILLED:
                blackboard.mark_needs_revalidation(dependent)
                invalidated.append(dependent)
                queue.append(dependent)  # cascade further downstream
    return sorted(invalidated)
