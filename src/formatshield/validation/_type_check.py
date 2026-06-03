"""Type and constraint validation for extracted field values (Layer 1).

This is Layer 1 of the three-layer validation stack:
- Layer 1 (this module): type + constraint validation — zero API calls.
- Layer 2 (post-MVP): GSV grounding-based semantic validation.
- Layer 3 (post-MVP): NLI natural-language inference validation.

All functions are pure and synchronous. The only I/O is regex compilation
at module load time (compiled once, reused).

Validation strategy
-------------------
1. Check for ``None`` / :data:`~formatshield.extraction._sfep.NEEDS_REVALIDATION`.
2. Attempt type validation via ``isinstance``.
3. On type mismatch, attempt coercion (e.g. ``"42"`` → ``int(42)``).
4. Check schema constraints (minLength, maxLength, minimum, maximum, pattern,
   enum, format).
5. Return ``(True, None)`` on success or ``(False, error_message)`` on failure.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

from formatshield.extraction._sfep import NEEDS_REVALIDATION

if TYPE_CHECKING:
    from formatshield.schema._types import Field

__all__ = [
    "constraint_check",
    "validate_field",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Basic format validators (MVP subset — not RFC-compliant, good enough for extraction)
_RE_EMAIL: re.Pattern[str] = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_RE_URI: re.Pattern[str] = re.compile(r"^https?://\S+$")
_RE_DATE: re.Pattern[str] = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RE_DATETIME: re.Pattern[str] = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")
_RE_TIME: re.Pattern[str] = re.compile(r"^\d{2}:\d{2}(:\d{2})?$")
_RE_UUID: re.Pattern[str] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

_FORMAT_VALIDATORS: dict[str, re.Pattern[str]] = {
    "email": _RE_EMAIL,
    "uri": _RE_URI,
    "date": _RE_DATE,
    "date-time": _RE_DATETIME,
    "time": _RE_TIME,
    "uuid": _RE_UUID,
}

# Sentinel imported at runtime for identity check (no circular import: validation does not
# import from assembly; extraction does not import from validation).


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_field(value: Any, field: Field) -> tuple[bool, str | None]:
    """Validate an extracted value against its field's type and constraints.

    Checks type first, then constraints. Attempts coercion before declaring
    a type mismatch to handle common LLM outputs like ``"42"`` for integer
    fields.

    Args:
        value: The extracted Python value (already typecast by the SFEP parser).
        field: Schema field descriptor with type and constraint metadata.

    Returns:
        ``(True, None)`` if valid. ``(False, error_message)`` if invalid.

    Example:
        >>> from formatshield.schema._types import Field
        >>> f = Field("age", "integer", {"minimum": 0}, "", {})
        >>> validate_field(25, f)
        (True, None)
        >>> validate_field(-5, f)
        (False, 'age: minimum constraint violated — -5 < 0')
        >>> validate_field("not_a_number", f)
        (False, "age: expected integer, got str 'not_a_number'")
    """
    # None is always valid — represents a missing optional field
    if value is None:
        return True, None

    # NEEDS_REVALIDATION sentinel — not a validation failure, handled by blackboard.
    # Use identity check (is), not repr() comparison — any object whose __repr__
    # returns "NEEDS_REVALIDATION" would otherwise bypass validation.
    if value is NEEDS_REVALIDATION:
        return True, None

    field_type = field.type

    # --- Type check with coercion fallback ---
    is_correct_type, coerced, type_error = _check_type(value, field_type, field)
    if not is_correct_type:
        return False, type_error
    # Use coerced value for constraint checks (e.g., int("42") = 42)
    effective_value = coerced if coerced is not None else value

    # --- Constraint checks ---
    violations = constraint_check(effective_value, field)
    if violations:
        return False, violations[0]

    return True, None


def constraint_check(value: Any, field: Field) -> list[str]:
    """Check all schema constraints for a value and return violation messages.

    Only checks constraints that are present in ``field.constraints``. Checks
    are applied in this order: minLength/maxLength, minimum/maximum,
    exclusiveMinimum/exclusiveMaximum, multipleOf, pattern, enum, format.

    Args:
        value: The value to check (must already be the correct Python type).
        field: Field descriptor providing the constraint dict.

    Returns:
        List of human-readable violation messages. Empty list if all pass.

    Example:
        >>> from formatshield.schema._types import Field
        >>> f = Field("code", "string", {"minLength": 3, "maxLength": 5}, "", {})
        >>> constraint_check("ab", f)
        ['code: minLength constraint violated — length 2 < 3']
        >>> constraint_check("abc", f)
        []
    """
    constraints = field.constraints
    violations: list[str] = []
    path = field.path

    # String length constraints
    if isinstance(value, str):
        if "minLength" in constraints:
            min_len = int(constraints["minLength"])
            if len(value) < min_len:
                violations.append(
                    f"{path}: minLength constraint violated — length {len(value)} < {min_len}"
                )
        if "maxLength" in constraints:
            max_len = int(constraints["maxLength"])
            if len(value) > max_len:
                violations.append(
                    f"{path}: maxLength constraint violated — length {len(value)} > {max_len}"
                )
        if "pattern" in constraints:
            pattern = str(constraints["pattern"])
            try:
                if not re.search(pattern, value):
                    violations.append(
                        f"{path}: pattern constraint violated — "
                        f"{value!r} does not match /{pattern}/"
                    )
            except re.error:
                violations.append(f"{path}: pattern constraint has invalid regex: {pattern!r}")
        if "format" in constraints:
            fmt = str(constraints["format"])
            fmt_re = _FORMAT_VALIDATORS.get(fmt)
            if fmt_re is not None and not fmt_re.match(value):
                violations.append(
                    f"{path}: format constraint violated — {value!r} is not a valid {fmt!r}"
                )

    # Numeric range constraints
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in constraints:
            minimum = constraints["minimum"]
            if value < minimum:
                violations.append(f"{path}: minimum constraint violated — {value} < {minimum}")
        if "maximum" in constraints:
            maximum = constraints["maximum"]
            if value > maximum:
                violations.append(f"{path}: maximum constraint violated — {value} > {maximum}")
        if "exclusiveMinimum" in constraints:
            ex_min = constraints["exclusiveMinimum"]
            if value <= ex_min:
                violations.append(
                    f"{path}: exclusiveMinimum constraint violated — {value} <= {ex_min}"
                )
        if "exclusiveMaximum" in constraints:
            ex_max = constraints["exclusiveMaximum"]
            if value >= ex_max:
                violations.append(
                    f"{path}: exclusiveMaximum constraint violated — {value} >= {ex_max}"
                )
        if "multipleOf" in constraints:
            multiple = constraints["multipleOf"]
            # Float-safe multipleOf check: divide and verify quotient is close to an integer.
            # Using value % multiple is unreliable (e.g. 0.3 % 0.1 = 0.09999... not 0).
            if multiple != 0:
                quotient = value / multiple
                if not math.isclose(quotient, round(quotient), rel_tol=1e-9):
                    violations.append(
                        f"{path}: multipleOf constraint violated — "
                        f"{value} is not a multiple of {multiple}"
                    )

    # Enum membership
    if "enum" in constraints:
        allowed = constraints["enum"]
        if value not in allowed:
            violations.append(f"{path}: enum constraint violated — {value!r} not in {allowed}")

    # Array item count
    if isinstance(value, list):
        if "minItems" in constraints:
            min_items = int(constraints["minItems"])
            if len(value) < min_items:
                violations.append(
                    f"{path}: minItems constraint violated — {len(value)} < {min_items}"
                )
        if "maxItems" in constraints:
            max_items = int(constraints["maxItems"])
            if len(value) > max_items:
                violations.append(
                    f"{path}: maxItems constraint violated — {len(value)} > {max_items}"
                )

    return violations


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_type(
    value: Any,
    field_type: str,
    field: Field,
) -> tuple[bool, Any, str | None]:
    """Check value against expected field type, attempting coercion if needed.

    Args:
        value: Value to check.
        field_type: Expected JSON Schema type string.
        field: Field descriptor for error message context.

    Returns:
        Tuple of ``(is_valid, coerced_value, error_message)``.
        ``coerced_value`` is ``None`` when no coercion was needed or possible.
    """
    path = field.path

    if field_type == "boolean":
        if isinstance(value, bool):
            return True, None, None
        # Coerce string representations
        if isinstance(value, str):
            lower = value.lower()
            if lower in ("true", "yes", "1"):
                return True, True, None
            if lower in ("false", "no", "0"):
                return True, False, None
        return False, None, f"{path}: expected boolean, got {type(value).__name__} {value!r}"

    if field_type == "integer":
        if isinstance(value, bool):
            # bool is a subtype of int in Python — reject
            return False, None, f"{path}: expected integer, got bool {value!r}"
        if isinstance(value, int):
            return True, None, None
        # Coerce string/float
        if isinstance(value, str):
            try:
                coerced = int(value)
                return True, coerced, None
            except ValueError:
                try:
                    as_float = float(value)
                    if as_float == int(as_float):
                        return True, int(as_float), None
                except ValueError:
                    pass
        if isinstance(value, float) and value == int(value):
            return True, int(value), None
        return False, None, f"{path}: expected integer, got {type(value).__name__} {value!r}"

    if field_type == "number":
        if isinstance(value, bool):
            return False, None, f"{path}: expected number, got bool {value!r}"
        if isinstance(value, (int, float)):
            return True, None, None
        if isinstance(value, str):
            try:
                coerced_number = float(value)
                return True, coerced_number, None
            except ValueError:
                pass
        return False, None, f"{path}: expected number, got {type(value).__name__} {value!r}"

    if field_type == "string":
        if isinstance(value, str):
            return True, None, None
        # Coerce common scalar types to string
        if isinstance(value, (int, float, bool)):
            return True, str(value), None
        return False, None, f"{path}: expected string, got {type(value).__name__} {value!r}"

    if field_type == "array":
        if isinstance(value, list):
            return True, None, None
        return False, None, f"{path}: expected array, got {type(value).__name__} {value!r}"

    if field_type == "null":
        if value is None:
            return True, None, None
        return False, None, f"{path}: expected null, got {type(value).__name__} {value!r}"

    if field_type == "enum":
        # Enum fields: any string value is type-valid; enum membership is a constraint
        if isinstance(value, str):
            return True, None, None
        return False, None, f"{path}: expected enum string, got {type(value).__name__} {value!r}"

    # object / unknown type — accept any value (schema structure handled upstream)
    return True, None, None
