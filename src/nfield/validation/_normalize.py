"""Value normalization run before validation.

Coerces common value forms to the schema type so a correct value is not rejected on
format alone - e.g. ``"$1,234,568" -> 1234568``, ``"Female" -> "Female"`` for an enum
cased ``"female"``. Keyed only on the JSON Schema ``type``/``constraints`` - never on
field name or domain. Lossless-or-decline: an ambiguous input is returned unchanged for
the validator to judge, never guessed. Bool set follows Pydantic v1 ``bool_validator``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nfield.schema._types import Field

__all__ = ["coerce_number", "normalize_value"]

# Pydantic v1 bool string sets (lowercased).
_BOOL_TRUE = frozenset({"1", "true", "t", "yes", "y", "on"})
_BOOL_FALSE = frozenset({"0", "false", "f", "no", "n", "off"})
# Symbols stripped from a number's ends (currency + whitespace).
_CURRENCY = "$€£¥₹₩ \t"
# Locale group separators: comma, dot, apostrophe variants, and space variants.
_GROUP_CHARS = ",.' " + chr(0x2019) + chr(0xA0) + chr(0x202F)
# Plain digits with an optional dot decimal (the canonical machine form).
_PLAIN_NUMBER = re.compile(r"^\d+(?:\.\d+)?$")


def normalize_value(value: Any, field: Field) -> Any:
    """Return *value* coerced to *field*'s type, or unchanged if it can't be (decline).

    Args:
        value: The extracted value (only strings are transformed; other types pass
            through so an already-canonical value is never altered).
        field: Schema field descriptor (drives the rule via ``type``/``constraints``).

    Returns:
        The normalized value, or the original when normalization is ambiguous or N/A.

    Example:
        >>> from nfield.schema._types import Field
        >>> f = Field("rev", "number", {}, "", {})
        >>> normalize_value("$1,234,568", f)
        1234568.0
        >>> normalize_value("nope", f)
        'nope'
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    ftype = field.type

    if ftype in ("integer", "number"):
        num = coerce_number(text)
        if num is None:
            return value
        if ftype == "integer":
            return int(num) if float(num).is_integer() else value
        return num

    if ftype == "boolean":
        low = text.lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        return value

    if ftype == "enum":
        allowed = field.constraints.get("enum")
        if isinstance(allowed, (list, tuple)):
            for option in allowed:
                if isinstance(option, str) and option.lower() == text.lower():
                    return option  # canonical casing
        return value

    if ftype == "string":
        return _strip_quotes(text)

    return value


def coerce_number(text: str) -> float | None:
    """Parse a formatted number across locales, or ``None`` when ambiguous.

    Handles currency symbols, sign, accounting parentheses, percent, and the common
    international grouping styles: US/Indian comma, European dot, Swiss apostrophe,
    and space (plain / no-break / narrow). When both a dot and a comma are present
    the last of the two is read as the decimal mark, so ``1,234.56`` and ``1.234,56``
    both parse to ``1234.56``. A single separator that could be either grouping or a
    decimal (e.g. ``1,23``) is declined rather than guessed. Shared by the validator
    and the SFEP parser so a formatted figure is never dropped on cast.

    Args:
        text: A stripped candidate string.

    Returns:
        The numeric value (sign applied), or ``None`` to decline.

    Example:
        >>> coerce_number("$1,234,568")
        1234568.0
        >>> coerce_number("1.234,56")
        1234.56
        >>> coerce_number("1,23") is None
        True
    """
    s = text
    negative = False
    if s.startswith("(") and s.endswith(")"):  # accounting negative
        negative = True
        s = s[1:-1].strip()
    if s.endswith("%"):  # keep the stated number; dividing by 100 could corrupt it
        s = s[:-1].strip()
    s = s.strip(_CURRENCY)
    if s[:1] in ("+", "-"):  # sign may sit before or after a currency symbol
        negative ^= s[0] == "-"
        s = s[1:].strip(_CURRENCY)
    canonical = _to_canonical_number(s)
    if canonical is None:
        return None
    return -canonical if negative else canonical


def _to_canonical_number(s: str) -> float | None:
    """Resolve grouping/decimal separators to a plain float, or ``None`` if ambiguous."""
    if _PLAIN_NUMBER.match(s):
        return float(s)
    has_dot = "." in s
    has_comma = "," in s
    if has_dot and has_comma:
        # The rightmost of the two is the decimal mark; the other is grouping.
        decimal = "." if s.rfind(".") > s.rfind(",") else ","
        grouping = "," if decimal == "." else "."
        body = s.replace(grouping, "").replace(decimal, ".")
        return float(body) if _PLAIN_NUMBER.match(body) else None
    for group_char in _GROUP_CHARS:
        if group_char in s and group_char not in ".,":
            # Apostrophe / space grouping is unambiguous; any remaining . is decimal.
            body = s.replace(group_char, "")
            return float(body) if _PLAIN_NUMBER.match(body) else None
    # A single separator counts only as strict 3-digit grouping; 1,23 declines.
    sep = "." if has_dot else ","
    if re.fullmatch(rf"\d{{1,3}}(?:\{sep}\d{{3}})+", s):
        return float(s.replace(sep, ""))
    return None


def _strip_quotes(text: str) -> str:
    """Drop one layer of surrounding matching quotes (an LLM output artifact).

    Args:
        text: A stripped string.

    Returns:
        The string without one pair of wrapping quotes, else unchanged.
    """
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text
