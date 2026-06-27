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
# A clean number body: plain digits, or 3-digit comma grouping, with one optional
# decimal. Grouping is validated by the regex so removing commas can't corrupt a
# European decimal like "1,23" (which fails to match and is declined).
_NUMBER_BODY = re.compile(r"^(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?$")


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
    """Parse a formatted number (currency/commas/percent/accounting parens), or None.

    Conservative: returns ``None`` for anything that isn't an unambiguous number, so
    the caller can reject or fall back rather than a guess being stored. Shared by the
    validator and the SFEP parser so a formatted figure is never dropped on cast.

    Args:
        text: A stripped candidate string.

    Returns:
        The numeric value (sign applied), or ``None`` to decline.

    Example:
        >>> coerce_number("$1,234,568")
        1234568.0
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
    if not _NUMBER_BODY.match(s):
        return None
    number = float(s.replace(",", ""))
    return -number if negative else number


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
