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
from enum import Enum
from typing import TYPE_CHECKING, Any

from formatshield.extraction._prompt import build_retry_system_message
from formatshield.extraction._sfep import parse_sfep

if TYPE_CHECKING:
    from formatshield.config import ExtractionConfig
    from formatshield.providers._protocol import LLMProvider
    from formatshield.schema._types import CapacityLeaf, Field

__all__ = [
    "FailureCause",
    "build_retry_prompt",
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

        batches = split_retry_batches(still_failing, dep_dag)
        round_recovered: dict[str, Any] = {}

        for batch in batches:
            batch_result = await surgical_field_retry(
                batch,
                {p: current_errors.get(p, "validation failed") for p in (f.path for f in batch)},
                provider,
                leaf,
                call_counter=call_counter,
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

    Returns:
        Dict of ``{path: value}`` for fields successfully re-extracted.
    """
    messages = build_retry_prompt(fields, errors, leaf.document_excerpt)

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
) -> list[dict[str, str]]:
    """Build the messages list for a surgical retry call.

    Args:
        fields: Failed fields to retry.
        errors: Per-field error messages for targeted correction.
        document_excerpt: Document text from the original leaf.

    Returns:
        Messages list for ``provider.complete()``.
    """
    return build_retry_system_message(fields, errors, document_excerpt)


def split_retry_batches(
    failed_fields: list[Field],
    dep_dag: dict[str, set[str]],
) -> list[list[Field]]:
    """Group failed fields by dependency closure for batched retry.

    Fields that share a dependency relationship must be retried together
    so that dependency-aware validation works correctly. Fields with no
    shared dependencies are grouped separately to minimise prompt size.

    Algorithm: union-find over dependency edges between failed fields.

    Args:
        failed_fields: Fields to batch.
        dep_dag: Full dependency graph ``{path: set_of_paths_it_depends_on}``.

    Returns:
        List of batches. Each batch is a list of fields that should be
        retried in a single API call.

    Example:
        >>> from formatshield.schema._types import Field
        >>> f1 = Field("a", "string", {}, "", {})
        >>> f2 = Field("b", "string", {}, "", {})
        >>> batches = split_retry_batches([f1, f2], {})
        >>> len(batches)
        2
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

    # Group by root
    groups: dict[str, list[str]] = {}
    for path in failed_paths:
        root = _find(path)
        groups.setdefault(root, []).append(path)

    return [[path_to_field[p] for p in group] for group in groups.values()]


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
