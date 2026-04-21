"""
Schema complexity-aware retry budget allocation and targeted failure correction.

Allocates retry attempts proportionally to schema complexity rather than
applying a fixed ``max_reasks=2`` regardless of schema structure.

Budget allocation is determined by three schema complexity signals:
  λ̃₂ (algebraic connectivity) — how coupled the field dependency graph is
  τ (constraint tightness)    — fraction of constrained fields (enums, booleans)
  field_count                  — total number of schema fields

Failure triage classifies validation errors into specific categories so that
each retry targets only the failing field(s) instead of replaying the full
generation:

  ENUM_VIOLATION     → 0.1 budget (single-token correction)
  RANGE_VIOLATION    → 0.2 budget (brief range clarification)
  TYPE_FAILURE       → 0.2 budget (brief type clarification)
  MISSING_FIELD      → 0.3 budget (surgical field injection)
  ARRAY_CARDINALITY  → 0.5 budget (array completion injection)
  CONSISTENCY_FAILURE→ 1.0 budget (routes to aggregation correction)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------


class FailureType(Enum):
    """Taxonomy of validation failure categories for targeted retry routing."""

    TYPE_FAILURE = "type"
    """LLM generated wrong type (string instead of boolean, etc.)."""

    CONSISTENCY_FAILURE = "consistency"
    """Field value inconsistent with a semantic aggregation constraint."""

    MISSING_FIELD = "missing"
    """Required field absent from the generated output."""

    ENUM_VIOLATION = "enum"
    """Generated value not in the allowed enum set."""

    ARRAY_CARDINALITY = "cardinality"
    """Array has wrong number of items (minItems or maxItems violated)."""

    RANGE_VIOLATION = "range"
    """Numeric field outside required bounds (minimum or maximum violated)."""


# ---------------------------------------------------------------------------
# Cost table (budget units consumed per retry attempt)
# ---------------------------------------------------------------------------

_FAILURE_COSTS: dict[FailureType, float] = {
    FailureType.ENUM_VIOLATION: 0.1,
    FailureType.RANGE_VIOLATION: 0.2,
    FailureType.TYPE_FAILURE: 0.2,
    FailureType.MISSING_FIELD: 0.3,
    FailureType.ARRAY_CARDINALITY: 0.5,
    FailureType.CONSISTENCY_FAILURE: 1.0,
}

_RECOMMENDED_ACTIONS: dict[FailureType, str] = {
    FailureType.ENUM_VIOLATION: "enum_constraint_tightening",
    FailureType.RANGE_VIOLATION: "range_clarification_reask",
    FailureType.TYPE_FAILURE: "type_clarification_reask",
    FailureType.MISSING_FIELD: "surgical_field_injection",
    FailureType.ARRAY_CARDINALITY: "array_completion_injection",
    FailureType.CONSISTENCY_FAILURE: "sac_correction",
}


# ---------------------------------------------------------------------------
# Public data contracts
# ---------------------------------------------------------------------------


@dataclass
class FailureClassification:
    """
    Classification of a single validation failure with retry routing info.

    Attributes
    ----------
    failure_type:
        Category of the failure.
    field_path:
        JSON path of the failing field (e.g. ``"score"`` or ``"(unknown)"``).
    error_message:
        Original validation error message.
    retry_cost:
        Budget units consumed when retrying this failure type.
    recommended_action:
        Routing key for retry strategy selection.
    """

    failure_type: FailureType
    field_path: str
    error_message: str
    retry_cost: float
    recommended_action: str

    def __post_init__(self) -> None:
        if self.retry_cost < 0:
            raise ValueError("retry_cost must be non-negative")

    @property
    def routes_to_sac(self) -> bool:
        """True when this failure should be routed to aggregation correction."""
        return self.failure_type == FailureType.CONSISTENCY_FAILURE


@dataclass
class BudgetAllocation:
    """
    Retry budget allocation result.

    Attributes
    ----------
    initial_budget:
        Total retry budget [0.0, 4.0] allocated based on schema complexity.
    lambda2:
        Algebraic connectivity used for allocation.
    tau:
        Constraint tightness used for allocation.
    field_count:
        Number of fields in schema used for allocation.
    """

    initial_budget: float
    lambda2: float
    tau: float
    field_count: int


# ---------------------------------------------------------------------------
# Budget allocator
# ---------------------------------------------------------------------------


class RetryBudgetAllocator:
    """
    Allocate retry budget proportionally to schema complexity.

    Uses algebraic connectivity (λ̃₂), constraint tightness (τ), and field
    count to determine how many retry budget units to allow per request.

    Parameters
    ----------
    schema:
        JSON Schema dict to analyze.
    lambda2:
        Algebraic connectivity of the schema dependency graph [0.0, 1.0].
        Values < 0.2 indicate flat/disconnected schemas.
    tau:
        Constraint tightness (fraction of constrained fields) [0.0, 1.0].
        Values > 0.7 indicate many precision-constrained fields.
    """

    def __init__(
        self,
        schema: dict[str, Any],
        lambda2: float = 0.5,
        tau: float = 0.5,
    ) -> None:
        self.schema = schema
        self.lambda2 = max(0.0, min(1.0, lambda2))
        self.tau = max(0.0, min(1.0, tau))
        self.field_count = _count_fields(schema)
        self._cached_budget: float | None = None

    def allocate(self) -> float:
        """
        Return retry budget [0.0, 4.0] based on schema complexity signals.

        Budget tiers by algebraic connectivity:
          λ̃₂ < 0.2 (flat):          1.0 (≤3 fields) or 1.5 (>3 fields)
          λ̃₂ 0.2–0.5 (moderate):    2.5
          λ̃₂ ≥ 0.5 (high coupling): 3.0 (≤10 fields) or 4.0 (>10 fields)

        Adjusted upward by 20% when τ > 0.7 (many constrained fields).
        Result capped at 4.0.
        """
        if self._cached_budget is not None:
            return self._cached_budget

        if self.lambda2 < 0.2:
            base = 1.0 if self.field_count <= 3 else 1.5
        elif self.lambda2 < 0.5:
            base = 2.5
        else:
            base = 3.0 if self.field_count <= 10 else 4.0

        if self.tau > 0.7:
            base *= 1.2

        self._cached_budget = min(base, 4.0)
        return self._cached_budget

    def allocation_info(self) -> BudgetAllocation:
        """Return structured allocation info including all input signals."""
        return BudgetAllocation(
            initial_budget=self.allocate(),
            lambda2=self.lambda2,
            tau=self.tau,
            field_count=self.field_count,
        )

    def cost_of_retry(self, failure_type: FailureType) -> float:
        """Return budget cost in [0.1, 1.0] for a retry of this failure type."""
        return _FAILURE_COSTS.get(failure_type, 1.0)

    def can_retry(self, remaining_budget: float, failure_type: FailureType) -> bool:
        """Return True when remaining budget covers a retry of this failure type."""
        return remaining_budget >= self.cost_of_retry(failure_type)


# ---------------------------------------------------------------------------
# Failure triager
# ---------------------------------------------------------------------------


class FailureTriager:
    """
    Classify validation errors into categories for targeted retry routing.

    Checks for semantic consistency failures first (using aggregation rules),
    then pattern-matches the validation error message for structural failures.
    """

    def classify(
        self,
        validation_error: str,
        output: dict[str, Any],
        aggregation_rules: list[Any] | None = None,
    ) -> FailureClassification:
        """
        Classify a single validation failure.

        Parameters
        ----------
        validation_error:
            Validation error message string (e.g. from jsonschema).
        output:
            Generated output dict that failed validation.
        aggregation_rules:
            Compiled aggregation rules from AggregationCompiler.
            When provided, consistency failures are detected first.

        Returns
        -------
        FailureClassification
            Failure type, field path, cost, and recommended action.
        """
        # ── 1. Consistency failure (semantic aggregation violation) ─────
        if aggregation_rules:
            from formatshield.reasoning.aggregation_compiler import (
                verify_aggregation_rules,
            )

            agg_result = verify_aggregation_rules(output, aggregation_rules)
            if not agg_result.passed:
                failed_fields = [rule.parent_field for rule, _ in agg_result.failed_rules]
                field_path = failed_fields[0] if failed_fields else "(aggregation)"
                return FailureClassification(
                    failure_type=FailureType.CONSISTENCY_FAILURE,
                    field_path=field_path,
                    error_message=validation_error,
                    retry_cost=_FAILURE_COSTS[FailureType.CONSISTENCY_FAILURE],
                    recommended_action=_RECOMMENDED_ACTIONS[FailureType.CONSISTENCY_FAILURE],
                )

        # ── 2. Missing required field ────────────────────────────────────
        missing_match = re.search(
            r"'([^']+)' is a required property|required property '([^']+)'",
            validation_error,
            re.IGNORECASE,
        )
        if missing_match:
            field_path = missing_match.group(1) or missing_match.group(2) or "(missing)"
            return FailureClassification(
                failure_type=FailureType.MISSING_FIELD,
                field_path=field_path,
                error_message=validation_error,
                retry_cost=_FAILURE_COSTS[FailureType.MISSING_FIELD],
                recommended_action=_RECOMMENDED_ACTIONS[FailureType.MISSING_FIELD],
            )

        # ── 3. Enum violation ────────────────────────────────────────────
        if re.search(
            r"is not one of|is not valid under any|enum",
            validation_error,
            re.IGNORECASE,
        ):
            field_path = _extract_field_from_error(validation_error)
            return FailureClassification(
                failure_type=FailureType.ENUM_VIOLATION,
                field_path=field_path,
                error_message=validation_error,
                retry_cost=_FAILURE_COSTS[FailureType.ENUM_VIOLATION],
                recommended_action=_RECOMMENDED_ACTIONS[FailureType.ENUM_VIOLATION],
            )

        # ── 4. Array cardinality ─────────────────────────────────────────
        if re.search(
            r"is too short|is too long|minItems|maxItems",
            validation_error,
            re.IGNORECASE,
        ):
            field_path = _extract_field_from_error(validation_error)
            return FailureClassification(
                failure_type=FailureType.ARRAY_CARDINALITY,
                field_path=field_path,
                error_message=validation_error,
                retry_cost=_FAILURE_COSTS[FailureType.ARRAY_CARDINALITY],
                recommended_action=_RECOMMENDED_ACTIONS[FailureType.ARRAY_CARDINALITY],
            )

        # ── 5. Range violation ───────────────────────────────────────────
        if re.search(
            r"(less|greater) than (the )?(minimum|maximum)",
            validation_error,
            re.IGNORECASE,
        ):
            field_path = _extract_field_from_error(validation_error)
            return FailureClassification(
                failure_type=FailureType.RANGE_VIOLATION,
                field_path=field_path,
                error_message=validation_error,
                retry_cost=_FAILURE_COSTS[FailureType.RANGE_VIOLATION],
                recommended_action=_RECOMMENDED_ACTIONS[FailureType.RANGE_VIOLATION],
            )

        # ── 6. Type failure (default fallback) ──────────────────────────
        field_path = _extract_field_from_error(validation_error)
        return FailureClassification(
            failure_type=FailureType.TYPE_FAILURE,
            field_path=field_path,
            error_message=validation_error,
            retry_cost=_FAILURE_COSTS[FailureType.TYPE_FAILURE],
            recommended_action=_RECOMMENDED_ACTIONS[FailureType.TYPE_FAILURE],
        )


# ---------------------------------------------------------------------------
# Surgical reasker
# ---------------------------------------------------------------------------


class SurgicalReasker:
    """
    Build targeted reask prompts that correct only the failing fields.

    Constructs a correction instruction showing:
    - Which specific fields need correction and their current (wrong) values
    - A specific fix description for each failure type
    - An explicit instruction to preserve all other output values unchanged

    This avoids replaying the full generation, which is both expensive and
    may introduce new errors in previously correct fields.
    """

    def build(
        self,
        output: dict[str, Any],
        failures: list[FailureClassification],
        original_prompt: str = "",
    ) -> str:
        """
        Build a surgical reask prompt targeting only the failing fields.

        Parameters
        ----------
        output:
            The generated output dict that failed validation.
        failures:
            List of FailureClassification instances from FailureTriager.
        original_prompt:
            Original generation prompt (unused; reserved for future context injection).

        Returns
        -------
        str
            Targeted reask instruction.  Empty string when failures is empty.
        """
        _ = original_prompt  # reserved for future context injection
        if not failures:
            return ""

        lines: list[str] = ["CORRECTION REQUIRED", "=" * 50, ""]

        lines.append("The following field(s) failed validation:\n")
        for failure in failures:
            current_val = output.get(failure.field_path, "<not present>")
            lines.append(
                f"  • '{failure.field_path}' (current: {current_val!r}): {failure.error_message}"
            )

        lines += ["", "=" * 50, ""]
        lines.append("Please UPDATE ONLY the listed field(s).")
        lines.append("Keep ALL other values unchanged.\n")

        for failure in failures:
            hint = _build_correction_hint(failure, output)
            if hint:
                lines.append(f"  → {hint}")

        lines.append("")
        lines.append("Return the complete corrected JSON object.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _count_fields(schema: dict[str, Any]) -> int:
    """Count total number of fields in a schema (recursive, depth-first)."""
    properties: dict[str, Any] = schema.get("properties") or {}
    count = len(properties)
    for fspec in properties.values():
        if not isinstance(fspec, dict):
            continue
        if fspec.get("type") == "object":
            count += _count_fields(fspec)
        elif fspec.get("type") == "array":
            items = fspec.get("items") or {}
            if isinstance(items, dict):
                count += _count_fields(items)
    return count


def _extract_field_from_error(error_message: str) -> str:
    """
    Attempt to extract a field name from a validation error message.

    Returns ``"(unknown)"`` when no field name can be identified.
    """
    # jsonschema style: "'field_name' is ..."
    match = re.search(r"'([^']+)'", error_message)
    if match:
        return match.group(1)
    # JSON path style: "$.field_name ..."
    match = re.search(r"\$\.([^\s]+)", error_message)
    if match:
        return match.group(1)
    return "(unknown)"


def _build_correction_hint(
    failure: FailureClassification,
    output: dict[str, Any],
) -> str:
    """Return a specific correction hint string for a failure classification."""
    f = failure.field_path

    if failure.failure_type == FailureType.MISSING_FIELD:
        return f"Add the required field '{f}' with an appropriate value."

    if failure.failure_type == FailureType.ENUM_VIOLATION:
        return (
            f"'{f}' must be one of the allowed enum values. "
            f"Current value {output.get(f)!r} is not in the allowed set."
        )

    if failure.failure_type == FailureType.RANGE_VIOLATION:
        return f"'{f}' must be within the schema's required numeric bounds."

    if failure.failure_type == FailureType.ARRAY_CARDINALITY:
        return (
            f"'{f}' must contain the required number of items "
            f"(check minItems / maxItems constraints)."
        )

    if failure.failure_type == FailureType.TYPE_FAILURE:
        return f"'{f}' has the wrong type. Check the schema for the required type."

    if failure.failure_type == FailureType.CONSISTENCY_FAILURE:
        return (
            f"'{f}' is inconsistent with related array items. "
            f"Recompute '{f}' from the array sub-field values — do not guess it independently."
        )

    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def allocate_retry_budget(
    schema: dict[str, Any],
    lambda2: float = 0.5,
    tau: float = 0.5,
) -> float:
    """
    Return retry budget [0.0, 4.0] based on schema complexity signals.

    Parameters
    ----------
    schema:
        JSON Schema dict.
    lambda2:
        Algebraic connectivity [0.0, 1.0].  Low = flat, high = coupled.
    tau:
        Constraint tightness [0.0, 1.0].  High = many enums/booleans.

    Returns
    -------
    float
        Retry budget.  Higher values mean more retry attempts are justified.
    """
    return RetryBudgetAllocator(schema, lambda2, tau).allocate()


def classify_failure(
    validation_error: str,
    output: dict[str, Any],
    aggregation_rules: list[Any] | None = None,
) -> FailureClassification:
    """
    Classify a validation failure for targeted retry routing.

    Parameters
    ----------
    validation_error:
        Validation error message string.
    output:
        Generated output dict that failed validation.
    aggregation_rules:
        Optional aggregation rules for consistency failure detection.

    Returns
    -------
    FailureClassification
        Failure category, cost, and recommended action.
    """
    return FailureTriager().classify(validation_error, output, aggregation_rules)


def build_surgical_reask(
    output: dict[str, Any],
    failures: list[FailureClassification],
    original_prompt: str = "",
) -> str:
    """
    Build a targeted reask prompt for only the failing fields.

    Parameters
    ----------
    output:
        The generated output dict that failed validation.
    failures:
        List of FailureClassification from classify_failure().
    original_prompt:
        Original generation prompt (optional).

    Returns
    -------
    str
        Targeted reask prompt.  Empty string when failures list is empty.
    """
    return SurgicalReasker().build(output, failures, original_prompt)
