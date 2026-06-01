from __future__ import annotations

import math
from typing import Any

from ._types import Field

__all__ = ["compute_tau"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_INTEGER_DIGITS: int = 5  # avg integer digit count
_DEFAULT_NUMBER_DIGITS: int = 6  # avg number digit count (before decimal)
_NUMBER_DECIMAL_OVERHEAD: int = 3  # decimal point + 2 decimals
_BOOL_TOKENS: int = 1
_NULL_TOKENS: int = 1
_DEFAULT_CHARS_PER_TOKEN: float = 4.0  # English average


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_tau(
    field: Field,
    chars_per_token: float,
    *,
    p90_string_tokens: int = 35,
    expected_array_size: int = 3,
) -> tuple[float, float]:
    """Compute expected output tokens (tau) and variance (var_tau) for a field.

    Implements SOTP (Schema-aware Output Token Predictor) rules per field type.
    These estimates are used for capacity planning (fits() in Stage 2C).

    Args:
        field: The Field to estimate tokens for.
        chars_per_token: Measured chars per token for the current model
            and language. Typically 3.5-4.5 for English.
        p90_string_tokens: 90th-percentile token count for unconstrained
            strings. Use domain-specific value from DomainConfig (default: 35).
        expected_array_size: Expected number of array elements for tau(array)
            calculation (default: 3).

    Returns:
        Tuple of (tau, var_tau) where:
        - tau: Expected output token count (>= 1)
        - var_tau: Variance of the token count estimate (>= 0)

    Example:
        >>> from formatshield.schema._types import Field
        >>> f = Field(path="active", type="boolean", constraints={},
        ...           parent_path="", schema_node={})
        >>> tau, var_tau = compute_tau(f, chars_per_token=4.0)
        >>> tau
        1.0
        >>> var_tau
        0.0
    """
    # Guard: chars_per_token must be positive
    if chars_per_token <= 0.0:
        chars_per_token = _DEFAULT_CHARS_PER_TOKEN

    tau: float
    var_tau: float

    ftype = field.type

    if ftype == "boolean":
        tau = float(_BOOL_TOKENS)
        var_tau = 0.0

    elif ftype == "null":
        tau = float(_NULL_TOKENS)
        var_tau = 0.0

    elif ftype == "enum":
        enum_values: list[Any] = field.constraints.get("enum", [""])
        tau = _compute_enum_tau(enum_values, chars_per_token)
        var_tau = 0.0

    elif ftype == "integer":
        tau = float(math.ceil(_DEFAULT_INTEGER_DIGITS / chars_per_token))
        var_tau = 0.5

    elif ftype == "number":
        total_chars = _DEFAULT_NUMBER_DIGITS + _NUMBER_DECIMAL_OVERHEAD
        tau = float(math.ceil(total_chars / chars_per_token))
        var_tau = 1.0

    elif ftype == "string":
        max_length = field.constraints.get("maxLength")
        if max_length is not None:
            tau = float(math.ceil(int(max_length) / chars_per_token))
            var_tau = (tau * 0.3) ** 2
        else:
            tau = float(p90_string_tokens)
            var_tau = (tau * 0.6) ** 2

    elif ftype == "array":
        items_info = field.schema_node.get("items")
        if isinstance(items_info, dict):
            # Build a temporary Field to recurse tau computation for element type
            element_type = items_info.get("type", "string")
            element_constraints = {
                k: v
                for k, v in items_info.items()
                if k in ("maxLength", "minimum", "maximum", "enum", "pattern", "format")
            }
            element_field = Field(
                path=field.path + "[]",
                type=element_type,
                constraints=element_constraints,
                parent_path=field.path,
                schema_node=items_info,
            )
            element_tau, element_var = compute_tau(
                element_field,
                chars_per_token,
                p90_string_tokens=p90_string_tokens,
                expected_array_size=expected_array_size,
            )
        else:
            element_tau = float(p90_string_tokens)
            element_var = (element_tau * 0.6) ** 2

        max_items = field.constraints.get("maxItems")
        array_size = int(max_items) if max_items is not None else expected_array_size

        tau = element_tau * array_size
        var_tau = element_var * array_size

    elif ftype == "object":
        tau = float(p90_string_tokens)
        var_tau = (tau * 0.6) ** 2

    else:
        # Unknown type — fallback to unconstrained string
        tau = float(p90_string_tokens)
        var_tau = (tau * 0.6) ** 2

    # Enforce minimum tau of 1.0
    tau = max(tau, 1.0)

    return (tau, var_tau)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_enum_tau(enum_values: list[Any], chars_per_token: float) -> float:
    """Compute tau for an enum field based on the longest option.

    Args:
        enum_values: List of enum values (any type).
        chars_per_token: Chars per token for current model/language.

    Returns:
        tau as float (>= 1.0).

    Example:
        >>> _compute_enum_tau(["USD", "EUR", "GBP"], 4.0)
        1.0
    """
    if not enum_values:
        return 1.0
    max_len = max(len(str(v)) for v in enum_values)
    tau = float(math.ceil(max_len / chars_per_token))
    return max(tau, 1.0)
