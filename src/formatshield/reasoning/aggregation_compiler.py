"""
Aggregation rule compiler for JSON schema structured generation.

Derives aggregation rules from JSON schema structure and enforces them
post-generation. When a parent field co-exists with an array whose items
share compatible types and semantically-related names, the parent's value
is *computed* from the array — not guessed independently.

Supported aggregation patterns:
  boolean  parent + array[boolean  items] → parent = ALL(items.boolean_field)
  numeric  parent + array[numeric  items] → parent = SUM or MEAN(items.numeric_field)
  enum     parent + array[same-enum items] → parent = "FAIL" if ANY item == "FAIL"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum string-overlap ratio to treat two field names as semantically
#: related (e.g. "valid" inside "argument_valid").
_NAME_OVERLAP_THRESHOLD: float = 0.5

#: Enum verdict values whose presence indicates a FAIL-if-any aggregation.
_FAIL_VERDICT_VALUES: frozenset[str] = frozenset(
    {"fail", "false", "invalid", "error", "rejected", "no", "0"}
)

#: Field name stems that indicate a boolean verdict role.
_VERDICT_STEMS: frozenset[str] = frozenset(
    {
        "valid",
        "correct",
        "match",
        "result",
        "pass",
        "success",
        "ok",
        "approved",
        "accepted",
        "sound",
    }
)


# ---------------------------------------------------------------------------
# Public data contracts
# ---------------------------------------------------------------------------


class AggregationPattern(Enum):
    """Detected aggregation relationship between parent and child fields."""

    BOOLEAN_ALL = "boolean_all"
    """parent = ALL(items.bool_field) — true only when every item is true."""

    NUMERIC_SUM = "numeric_sum"
    """parent = SUM(items.numeric_field)."""

    NUMERIC_MEAN = "numeric_mean"
    """parent = MEAN(items.numeric_field)."""

    ENUM_FAIL_ANY = "enum_fail_any"
    """parent = FAIL if ANY item has a fail-class value, else PASS."""


@dataclass
class AggregationRule:
    """
    A single derived aggregation rule between a parent field and an array.

    Attributes
    ----------
    parent_field:
        Top-level field whose value is computed (e.g. ``"argument_valid"``).
    array_field:
        Array field whose items supply the source values (e.g. ``"premises"``).
    child_field:
        Name of the sub-field within each array item to aggregate from.
    pattern:
        Detected aggregation pattern (boolean ALL, numeric SUM/MEAN, enum
        FAIL-if-any).
    execution_step_instruction:
        Ready-to-inject instruction string for the execution protocol.
    verification_description:
        Human-readable description of the post-generation check.
    fail_values:
        For ENUM_FAIL_ANY patterns: which enum values are treated as failures.
    """

    parent_field: str
    array_field: str
    child_field: str
    pattern: AggregationPattern
    execution_step_instruction: str
    verification_description: str
    fail_values: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.parent_field:
            raise ValueError("parent_field must not be empty")
        if not self.array_field:
            raise ValueError("array_field must not be empty")
        if not self.child_field:
            raise ValueError("child_field must not be empty")
        if not self.execution_step_instruction:
            raise ValueError("execution_step_instruction must not be empty")


@dataclass
class AggregationVerificationResult:
    """Result of post-generation aggregation verification."""

    passed: bool
    failed_rules: list[tuple[AggregationRule, str]]
    """List of (rule, reason) for every rule that failed verification."""

    def is_consistent(self) -> bool:
        """Return True when all aggregation rules are satisfied."""
        return self.passed


# ---------------------------------------------------------------------------
# Core compiler
# ---------------------------------------------------------------------------


class AggregationCompiler:
    """
    Detect and compile aggregation rules from a JSON schema.

    Detection operates in two phases:
    1. Collect boolean/numeric/enum *parent candidates* at the top level.
    2. Collect matching *child fields* inside array item schemas.
    3. For each (parent, child) pair that passes name-overlap + type-compat
       checks, emit an AggregationRule with the derived pattern.
    """

    def compile(self, schema: dict[str, Any]) -> list[AggregationRule]:
        """
        Derive all aggregation rules from schema structure.

        Parameters
        ----------
        schema:
            JSON Schema dict (top-level object schema).

        Returns
        -------
        list[AggregationRule]
            Ordered list of detected aggregation rules. Empty when the schema
            has no aggregatable patterns.
        """
        properties: dict[str, Any] = schema.get("properties") or {}
        rules: list[AggregationRule] = []

        # Separate top-level fields by type
        bool_parents = {
            name: spec
            for name, spec in properties.items()
            if isinstance(spec, dict) and spec.get("type") == "boolean"
        }
        numeric_parents = {
            name: spec
            for name, spec in properties.items()
            if isinstance(spec, dict) and spec.get("type") in ("number", "integer")
        }
        enum_parents = {
            name: spec
            for name, spec in properties.items()
            if isinstance(spec, dict) and "enum" in spec and spec.get("type") not in ("array", None)
        }
        array_fields = {
            name: spec
            for name, spec in properties.items()
            if isinstance(spec, dict) and spec.get("type") == "array"
        }

        for array_name, array_spec in array_fields.items():
            item_spec: dict[str, Any] = array_spec.get("items") or {}
            if not isinstance(item_spec, dict):
                continue
            item_props: dict[str, Any] = item_spec.get("properties") or {}
            if not item_props:
                continue

            # ── Boolean aggregation ──────────────────────────────────────
            for child_name, child_spec in item_props.items():
                if not isinstance(child_spec, dict):
                    continue
                if child_spec.get("type") == "boolean":
                    for parent_name in bool_parents:
                        if _names_are_related(parent_name, child_name):
                            rules.append(
                                _make_boolean_all_rule(parent_name, array_name, child_name)
                            )

            # ── Numeric aggregation ──────────────────────────────────────
            for child_name, child_spec in item_props.items():
                if not isinstance(child_spec, dict):
                    continue
                if child_spec.get("type") in ("number", "integer"):
                    for parent_name in numeric_parents:
                        if _names_are_related(parent_name, child_name):
                            pattern = _detect_numeric_pattern(parent_name)
                            rules.append(
                                _make_numeric_rule(
                                    parent_name,
                                    array_name,
                                    child_name,
                                    pattern,
                                )
                            )

            # ── Enum FAIL-if-any aggregation ────────────────────────────
            for child_name, child_spec in item_props.items():
                if not isinstance(child_spec, dict):
                    continue
                child_enum: list[Any] = child_spec.get("enum") or []
                if not child_enum:
                    continue
                for parent_name, parent_spec in enum_parents.items():
                    parent_enum: list[Any] = parent_spec.get("enum") or []
                    if not parent_enum:
                        continue
                    if _enums_share_fail_class(parent_enum, child_enum):
                        fail_vals = _detect_fail_values(parent_enum)
                        if fail_vals:
                            rules.append(
                                _make_enum_fail_any_rule(
                                    parent_name,
                                    array_name,
                                    child_name,
                                    fail_vals,
                                )
                            )

        return rules


# ---------------------------------------------------------------------------
# Rule factories (private helpers)
# ---------------------------------------------------------------------------


def _make_boolean_all_rule(parent: str, array_name: str, child: str) -> AggregationRule:
    """Build a BOOLEAN_ALL aggregation rule."""
    instruction = (
        f"★ DERIVED FIELD — '{parent}' MUST equal ALL({array_name}[*].{child}). "
        f"Do NOT guess '{parent}' independently. "
        f"If ANY item in '{array_name}' has {child}=false, then '{parent}' MUST be false. "
        f"Only set '{parent}' to true when EVERY item has {child}=true."
    )
    verification = (
        f"'{parent}' must equal ALL(item.{child} for item in {array_name}). "
        f"A {parent}=true when any {array_name} item has {child}=false is a consistency error."
    )
    return AggregationRule(
        parent_field=parent,
        array_field=array_name,
        child_field=child,
        pattern=AggregationPattern.BOOLEAN_ALL,
        execution_step_instruction=instruction,
        verification_description=verification,
    )


def _make_numeric_rule(
    parent: str,
    array_name: str,
    child: str,
    pattern: AggregationPattern,
) -> AggregationRule:
    """Build a NUMERIC_SUM or NUMERIC_MEAN aggregation rule."""
    if pattern == AggregationPattern.NUMERIC_SUM:
        op = "SUM"
        formula = f"sum(item.{child} for item in {array_name})"
    else:
        op = "MEAN"
        formula = f"mean(item.{child} for item in {array_name})"

    instruction = (
        f"★ DERIVED FIELD — '{parent}' MUST be computed as {op}({array_name}[*].{child}). "
        f"Do NOT estimate '{parent}' without calculating from the array. "
        f"Formula: {formula}."
    )
    verification = (
        f"'{parent}' must equal {formula}. "
        f"Manually assigned values that differ from the computed {op} are errors."
    )
    return AggregationRule(
        parent_field=parent,
        array_field=array_name,
        child_field=child,
        pattern=pattern,
        execution_step_instruction=instruction,
        verification_description=verification,
    )


def _make_enum_fail_any_rule(
    parent: str,
    array_name: str,
    child: str,
    fail_values: list[str],
) -> AggregationRule:
    """Build an ENUM_FAIL_ANY aggregation rule."""
    fail_repr = ", ".join(repr(v) for v in fail_values)
    instruction = (
        f"★ DERIVED FIELD — '{parent}' is determined by '{array_name}[*].{child}'. "
        f"Set '{parent}' to a fail value ({fail_repr}) if ANY item in '{array_name}' "
        f"has {child} in ({fail_repr}). "
        f"Only use a pass value when ALL items have non-fail {child} values."
    )
    verification = (
        f"'{parent}' must be a fail value if any {array_name} item has "
        f"{child} in {{{fail_repr}}}. Inconsistent parent/child verdicts are errors."
    )
    return AggregationRule(
        parent_field=parent,
        array_field=array_name,
        child_field=child,
        pattern=AggregationPattern.ENUM_FAIL_ANY,
        execution_step_instruction=instruction,
        verification_description=verification,
        fail_values=fail_values,
    )


# ---------------------------------------------------------------------------
# Post-generation verifier
# ---------------------------------------------------------------------------


def verify_aggregation_rules(
    output: dict[str, Any],
    rules: list[AggregationRule],
) -> AggregationVerificationResult:
    """
    Verify that a generated output satisfies all aggregation rules.

    Parameters
    ----------
    output:
        Parsed JSON output dict from Pass 2 generation.
    rules:
        Aggregation rules compiled from the schema by AggregationCompiler.

    Returns
    -------
    AggregationVerificationResult
        passed=True when all rules are satisfied; failed_rules lists any
        (rule, reason) pairs where the output is inconsistent.
    """
    failed: list[tuple[AggregationRule, str]] = []

    for rule in rules:
        parent_val = output.get(rule.parent_field)
        array_val = output.get(rule.array_field)

        if parent_val is None or array_val is None:
            # Cannot verify if either field is absent — skip gracefully
            continue
        if not isinstance(array_val, list):
            continue

        reason = _check_rule(rule, parent_val, array_val)
        if reason:
            failed.append((rule, reason))

    return AggregationVerificationResult(
        passed=len(failed) == 0,
        failed_rules=failed,
    )


def _check_rule(
    rule: AggregationRule,
    parent_val: Any,
    array_val: list[Any],
) -> str | None:
    """
    Check a single rule against parent and array values.

    Returns None when the rule is satisfied, or an error description string.
    """
    child = rule.child_field

    if rule.pattern == AggregationPattern.BOOLEAN_ALL:
        child_values = [
            item.get(child) for item in array_val if isinstance(item, dict) and child in item
        ]
        if not child_values:
            return None
        expected = all(bool(v) for v in child_values)
        if bool(parent_val) != expected:
            return (
                f"'{rule.parent_field}'={parent_val!r} but "
                f"ALL({rule.array_field}[*].{child})={expected}. "
                f"Expected parent to be {expected}."
            )

    elif rule.pattern == AggregationPattern.NUMERIC_SUM:
        child_values = [
            item.get(child) for item in array_val if isinstance(item, dict) and child in item
        ]
        numeric = [v for v in child_values if isinstance(v, (int, float))]
        if not numeric:
            return None
        expected_sum = sum(numeric)
        try:
            actual = float(parent_val)
        except (TypeError, ValueError):
            return (
                f"'{rule.parent_field}'={parent_val!r} is not numeric; expected SUM={expected_sum}."
            )
        if abs(actual - expected_sum) > 1e-6:
            return (
                f"'{rule.parent_field}'={actual} but "
                f"SUM({rule.array_field}[*].{child})={expected_sum}."
            )

    elif rule.pattern == AggregationPattern.NUMERIC_MEAN:
        child_values = [
            item.get(child) for item in array_val if isinstance(item, dict) and child in item
        ]
        numeric = [v for v in child_values if isinstance(v, (int, float))]
        if not numeric:
            return None
        expected_mean = sum(numeric) / len(numeric)
        try:
            actual = float(parent_val)
        except (TypeError, ValueError):
            return (
                f"'{rule.parent_field}'={parent_val!r} is not numeric; "
                f"expected MEAN={expected_mean:.4f}."
            )
        if abs(actual - expected_mean) > 1e-4:
            return (
                f"'{rule.parent_field}'={actual} but "
                f"MEAN({rule.array_field}[*].{child})={expected_mean:.4f}."
            )

    elif rule.pattern == AggregationPattern.ENUM_FAIL_ANY:
        child_values = [
            item.get(child) for item in array_val if isinstance(item, dict) and child in item
        ]
        fail_set = {v.lower() if isinstance(v, str) else v for v in rule.fail_values}
        any_failed = any(
            (str(v).lower() if isinstance(v, str) else v) in fail_set for v in child_values
        )
        parent_norm = str(parent_val).lower() if isinstance(parent_val, str) else parent_val
        parent_is_fail = parent_norm in fail_set
        if any_failed and not parent_is_fail:
            return (
                f"'{rule.parent_field}'={parent_val!r} indicates pass, but at least one "
                f"{rule.array_field} item has {child} in fail-class "
                f"{{{', '.join(repr(v) for v in rule.fail_values)}}}."
            )

    return None


# ---------------------------------------------------------------------------
# Surgical reask builder
# ---------------------------------------------------------------------------


def build_aggregation_reask(
    output: dict[str, Any],
    failed_rules: list[tuple[AggregationRule, str]],
) -> str:
    """
    Build a surgical reask prompt targeting only inconsistent derived fields.

    Instead of replaying the entire generation (expensive, often introduces
    new errors), this constructs a focused correction request.

    Parameters
    ----------
    output:
        The original output dict that failed verification. Used to show
        the current (incorrect) value in the reask for context.
    failed_rules:
        List of (AggregationRule, reason) from verify_aggregation_rules().

    Returns
    -------
    str
        A reask instruction ready to prepend to the next generation prompt.
    """
    if not failed_rules:
        return ""

    lines: list[str] = [
        "CORRECTION REQUIRED — the following derived fields are inconsistent:",
        "",
    ]

    for rule, reason in failed_rules:
        current_val = output.get(rule.parent_field, "<missing>")
        lines.append(f"• Field '{rule.parent_field}' (current value: {current_val!r}): {reason}")
        lines.append(f"  Fix: {rule.verification_description}")
        lines.append(f"  Rule: {rule.execution_step_instruction}")
        lines.append("")

    lines.append("Update ONLY the listed field(s). Keep all other output values unchanged.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Name-similarity helpers
# ---------------------------------------------------------------------------


def _names_are_related(parent_name: str, child_name: str) -> bool:
    """
    Determine whether two field names are semantically related.

    Checks:
    1. Direct containment: one name appears as a substring in the other.
    2. Stem match: both names share a common meaningful stem from
       _VERDICT_STEMS.
    3. Token overlap: shared word tokens after splitting on underscore/camel.
    """
    p_lower = parent_name.lower()
    c_lower = child_name.lower()

    # Direct containment (e.g. "argument_valid" contains "valid")
    if c_lower in p_lower or p_lower in c_lower:
        return True

    # Shared verdict stem (e.g. both contain "pass" or "valid")
    for stem in _VERDICT_STEMS:
        if stem in p_lower and stem in c_lower:
            return True

    # Token overlap (e.g. "total_score" and "score" share "score")
    p_tokens = set(_tokenize(parent_name))
    c_tokens = set(_tokenize(child_name))
    if p_tokens and c_tokens:
        overlap = len(p_tokens & c_tokens) / max(len(p_tokens), len(c_tokens))
        if overlap >= _NAME_OVERLAP_THRESHOLD:
            return True

    return False


def _tokenize(name: str) -> list[str]:
    """Split a field name into meaningful tokens (underscore and camelCase)."""
    import re

    # Split on underscore or camelCase boundaries
    tokens = re.sub(r"([A-Z])", r"_\1", name).lower().split("_")
    return [t for t in tokens if len(t) > 1]


def _detect_numeric_pattern(parent_name: str) -> AggregationPattern:
    """
    Heuristically detect whether the numeric parent is a SUM or MEAN.

    Rules:
    - "total", "sum", "count" in parent name → SUM
    - "average", "mean", "avg", "rate" in parent name → MEAN
    - Otherwise: SUM (safer default — overcounting is more visible)
    """
    p = parent_name.lower()
    if any(kw in p for kw in ("total", "sum", "count", "cumul")):
        return AggregationPattern.NUMERIC_SUM
    if any(kw in p for kw in ("mean", "avg", "average", "rate", "ratio")):
        return AggregationPattern.NUMERIC_MEAN
    return AggregationPattern.NUMERIC_SUM


def _enums_share_fail_class(parent_enum: list[Any], child_enum: list[Any]) -> bool:
    """
    Return True if both enum sets contain at least one fail-class value.

    This prevents spurious FAIL-if-any rules when two unrelated enum fields
    happen to co-exist with an array.
    """
    p_lower = {str(v).lower() for v in parent_enum}
    c_lower = {str(v).lower() for v in child_enum}
    return bool(p_lower & _FAIL_VERDICT_VALUES) and bool(c_lower & _FAIL_VERDICT_VALUES)


def _detect_fail_values(enum_values: list[Any]) -> list[str]:
    """
    Extract the fail-class values from an enum definition.

    Only string values that match _FAIL_VERDICT_VALUES are returned
    (case-insensitive, original case preserved).
    """
    return [str(v) for v in enum_values if str(v).lower() in _FAIL_VERDICT_VALUES]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_aggregation_rules(schema: dict[str, Any]) -> list[AggregationRule]:
    """
    Derive all aggregation rules from schema structure.

    This is the main entry point for the Semantic Aggregation Compiler.
    Call this once per schema and cache the result — schema analysis is
    deterministic and inexpensive.

    Parameters
    ----------
    schema:
        JSON Schema dict (top-level object with ``"properties"``).

    Returns
    -------
    list[AggregationRule]
        All detected aggregation rules in detection order.
        Empty when no aggregatable patterns are found.
    """
    return AggregationCompiler().compile(schema)
