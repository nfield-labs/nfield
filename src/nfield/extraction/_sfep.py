"""SFEP (Schema-aware Field Extraction Protocol) parser.

Instead of asking the LLM to output nested JSON - whose braces, quotes, commas,
and repeated keys spend a large share of tokens on structure, and whose
constrained decoding measurably degrades reasoning accuracy (arXiv:2408.02442;
arXiv:2604.03616, "The Format Tax") - the LLM outputs one field per line::

    field.path = value

This sidesteps that format tax and preserves a bijective mapping to nested JSON
(every field maps to exactly one path and back).

Parsing rules
-------------
* Each line is split on `` = `` (first occurrence only).
* Left side must be a dot-notation path matching a known field.
* Right side is a raw string value, cast to the field's Python type.
* ``NULL`` (case-insensitive) maps to Python ``None``.
* ``NEEDS_REVALIDATION`` maps to the :data:`NEEDS_REVALIDATION` sentinel.
* Array values are JSON arrays on one line (a multi-line array is accumulated
  until brackets balance); bare bracket lists ``[a, b, c]`` are also accepted.
* Unknown paths are silently skipped (LLM may hallucinate paths).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import json_repair

from nfield.exceptions import ExtractionError
from nfield.validation._normalize import coerce_number

if TYPE_CHECKING:
    from nfield.schema._types import Field

__all__ = [
    "NEEDS_REVALIDATION",
    "count_unknown_paths",
    "parse_sfep",
    "parse_sfep_failures",
    "parse_sfep_line",
    "typecast",
    "unclean_json_arrays",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SFEP_SEPARATOR: str = " = "
_SFEP_NULL_SENTINEL: str = "NULL"
_SFEP_NEEDS_REVALIDATION_SENTINEL: str = "NEEDS_REVALIDATION"
# Currency symbols and thousands-grouping separators that mark a displayed number.
_CURRENCY_SIGNS: str = "$" + chr(0x20AC) + chr(0xA3) + chr(0xA5) + chr(0x20B9) + chr(0x20A9)
_GROUP_SIGNS: str = ",.' " + chr(0x2019) + chr(0xA0) + chr(0x202F)


# ---------------------------------------------------------------------------
# NEEDS_REVALIDATION singleton sentinel
# ---------------------------------------------------------------------------


class _NeedsRevalidationType:
    """Singleton sentinel indicating a field flagged for revalidation by the LLM.

    The LLM outputs ``NEEDS_REVALIDATION`` when it found the field but cannot
    confidently determine the value. The blackboard state machine transitions
    the field to ``NEEDS_REVALIDATION`` state instead of ``FILLED``.

    Example:
        >>> from nfield.extraction._sfep import NEEDS_REVALIDATION
        >>> NEEDS_REVALIDATION is NEEDS_REVALIDATION
        True
    """

    _instance: _NeedsRevalidationType | None = None

    def __new__(cls) -> _NeedsRevalidationType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NEEDS_REVALIDATION"

    def __bool__(self) -> bool:
        return False


NEEDS_REVALIDATION: _NeedsRevalidationType = _NeedsRevalidationType()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_sfep(text: str, fields: list[Field]) -> dict[str, Any]:
    """Parse SFEP key=value output into a typed Python dict.

    Processes each line of *text* as a ``path = value`` pair. Values are
    typecast using the corresponding :class:`~nfield.schema._types.Field`
    descriptor. Lines that do not match any known field path are silently
    skipped.

    Args:
        text: Raw LLM output in SFEP format. May include leading/trailing
            whitespace and blank lines.
        fields: List of schema fields used for type resolution. Each field's
            ``path`` attribute is used as the lookup key.

    Returns:
        Dict mapping dot-notation field paths to typed Python values.
        :data:`NEEDS_REVALIDATION` is returned as a value when the LLM
        signals it cannot confidently extract the field.

    Example:
        >>> from nfield.schema._types import Field
        >>> f_name = Field("name", "string", {}, "", {})
        >>> f_age = Field("age", "integer", {}, "", {})
        >>> f_active = Field("active", "boolean", {}, "", {})
        >>> result = parse_sfep(
        ...     "name = Alice\\nage = 30\\nactive = true",
        ...     [f_name, f_age, f_active],
        ... )
        >>> result == {"name": "Alice", "age": 30, "active": True}
        True
    """
    field_map: dict[str, Field] = {f.path: f for f in fields}
    result: dict[str, Any] = {}

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        pair = parse_sfep_line(lines[i])
        if pair is None:
            i += 1
            continue
        path, raw_value = pair
        # Capture a multi-line JSON value whole (see _accumulate_value).
        raw_value, i = _accumulate_value(raw_value, lines, i + 1, field_map)
        field = field_map.get(path)
        if field is None:
            continue
        try:
            result[path] = typecast(raw_value, field)
        except ExtractionError:
            # Malformed value that can't be coerced - skip, blackboard handles missing
            continue

    return result


def _is_balanced(text: str) -> bool:
    """Return ``True`` when brackets/braces in *text* are balanced (ignoring strings)."""
    depth = 0
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
    return depth <= 0


def _accumulate_value(
    raw_value: str, lines: list[str], start: int, field_map: dict[str, Field]
) -> tuple[str, int]:
    """Join continuation lines of a multi-line bracketed value; return (value, next_index).

    Only a value that opens with ``[`` or ``{`` and is not yet balanced pulls in
    following lines. Accumulation stops at balance, at end of input, or at the next
    line that is itself a known ``field.path = value`` pair (so a later field is not
    swallowed by a truncated array).
    """
    if raw_value[:1] not in "[{" or _is_balanced(raw_value):
        return raw_value, start
    buffer = [raw_value]
    j = start
    while j < len(lines) and not _is_balanced("\n".join(buffer)):
        nxt = parse_sfep_line(lines[j])
        if nxt is not None and nxt[0] in field_map:
            break
        buffer.append(lines[j])
        j += 1
    return "\n".join(buffer), j


def count_unknown_paths(text: str, fields: list[Field]) -> int:
    """Count SFEP lines whose ``path`` is not a known schema field.

    A line that parses as ``path = value`` but whose path is absent from *fields* is the
    model emitting a field outside the schema - a format-drift / hallucination signal
    (analogous to a strict schema's "forbid extra" rule). Unparseable lines (no separator) are *not*
    counted: those are prose/noise, not invented fields. Extraction is unaffected; this is
    a measurement only.

    Args:
        text: Raw LLM output in SFEP format.
        fields: The schema fields the call requested (the known paths).

    Returns:
        The number of parseable lines whose path is not in *fields*.

    Example:
        >>> from nfield.schema._types import Field
        >>> f = Field("name", "string", {}, "", {})
        >>> count_unknown_paths("name = Alice\\nfavorite_color = blue", [f])
        1
    """
    known = {f.path for f in fields}
    unknown = 0
    for line in text.splitlines():
        pair = parse_sfep_line(line)
        if pair is not None and pair[0] not in known:
            unknown += 1
    return unknown


def unclean_json_arrays(text: str, fields: list[Field]) -> set[str]:
    """List-leaf array paths whose JSON value did not parse cleanly (repair was needed).

    Covers both object arrays and scalar list-leaves - any array carrying an item
    schema. A cleanly-parsed array is trustworthy; one that only survived repair (or
    a comma-split fallback) may be partial or shattered, so the caller can re-sample
    that leaf and keep the better result. Empty/absent values are not unclean.
    """
    field_map = {f.path: f for f in fields}
    unclean: set[str] = set()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        pair = parse_sfep_line(lines[i])
        if pair is None:
            i += 1
            continue
        path, raw = pair
        raw, i = _accumulate_value(raw, lines, i + 1, field_map)
        field = field_map.get(path)
        if (
            field is None
            or field.type != "array"
            or not isinstance(field.constraints.get("items"), dict)
        ):
            continue
        value = raw.strip()
        if not value or value in {"[]", "null", "none"}:
            continue
        start, end = value.find("["), value.rfind("]")
        if start == -1 or end <= start:
            unclean.add(path)
            continue
        try:
            json.loads(value[start : end + 1])
        except json.JSONDecodeError:
            unclean.add(path)
    return unclean


def parse_sfep_failures(text: str, fields: list[Field]) -> dict[str, str]:
    """Return the raw value of each known-path line whose typecast failed.

    ``parse_sfep`` drops a value it cannot coerce (e.g. ``age = abc`` for an integer).
    This scan keeps the raw string so the recovery prompt can show the model what it
    produced (DSPy Assertions, arXiv:2312.13382). Only genuine cast failures appear -
    NULL, NEEDS_REVALIDATION, empty, unknown paths, and clean casts are excluded.

    Args:
        text: Raw LLM output in SFEP format.
        fields: The schema fields the call requested.

    Returns:
        Dict mapping field path to the raw (uncast) string for each cast failure.

    Example:
        >>> from nfield.schema._types import Field
        >>> f = Field("age", "integer", {}, "", {})
        >>> parse_sfep_failures("age = abc", [f])
        {'age': 'abc'}
    """
    field_map: dict[str, Field] = {f.path: f for f in fields}
    failures: dict[str, str] = {}
    for line in text.splitlines():
        pair = parse_sfep_line(line)
        if pair is None:
            continue
        path, raw_value = pair
        field = field_map.get(path)
        if field is None:
            continue
        try:
            typecast(raw_value, field)
        except ExtractionError:
            failures[path] = raw_value
    return failures


def parse_sfep_line(line: str) -> tuple[str, str] | None:
    """Parse a single SFEP line into a (path, raw_value) pair.

    Splits on the first occurrence of `` = `` (space-equals-space). This
    separator was chosen to minimise false splits on values that contain ``=``
    (e.g. URLs, equations).

    Args:
        line: A single line of SFEP output, possibly with leading/trailing
            whitespace.

    Returns:
        ``(path, raw_value)`` tuple if the line is a valid SFEP pair,
        or ``None`` for blank lines, comment lines, and lines without
        the separator.

    Example:
        >>> parse_sfep_line("address.city = New York")
        ('address.city', 'New York')
        >>> parse_sfep_line("not a valid line") is None
        True
        >>> parse_sfep_line("") is None
        True
    """
    # Strip leading whitespace only - preserve trailing spaces in value
    lstripped = line.lstrip()
    if not lstripped or lstripped.startswith("#"):
        return None

    sep_idx = lstripped.find(_SFEP_SEPARATOR)
    if sep_idx == -1:
        return None

    path = lstripped[:sep_idx].strip()
    # Value: everything after the separator, strip only line-ending characters
    raw_value = lstripped[sep_idx + len(_SFEP_SEPARATOR) :].rstrip("\r\n")

    if not path:
        return None

    return path, raw_value


def typecast(raw_value: str, field: Field) -> Any:
    """Cast a raw SFEP string value to the Python type specified by *field*.

    Args:
        raw_value: Raw string from SFEP output (right-hand side of ``=``).
        field: Schema field descriptor providing type and constraint metadata.

    Returns:
        Typed Python value. Returns :data:`NEEDS_REVALIDATION` sentinel when
        the LLM explicitly signalled uncertainty. Returns ``None`` for NULL.

    Raises:
        ExtractionError: If the raw value cannot be cast to the expected type
            and coercion fails.

    Example:
        >>> from nfield.schema._types import Field
        >>> f = Field("count", "integer", {}, "", {})
        >>> typecast("42", f)
        42
        >>> f_bool = Field("active", "boolean", {}, "", {})
        >>> typecast("true", f_bool)
        True
    """
    stripped = raw_value.strip()

    # Universal sentinels - checked before type-specific logic.
    # NULL maps to None for all types.
    # An empty value maps to None for non-string types; for string fields the
    # empty string is a legitimate value and is returned as-is.
    if stripped.upper() == _SFEP_NULL_SENTINEL:
        return None
    if stripped == _SFEP_NEEDS_REVALIDATION_SENTINEL:
        return NEEDS_REVALIDATION
    if not stripped and field.type != "string":
        return None

    field_type = field.type

    if field_type == "null":
        return None

    if field_type == "boolean":
        return _cast_boolean(stripped, field)

    if field_type == "integer":
        return _cast_integer(stripped, field)

    if field_type == "number":
        return _cast_number(stripped, field)

    if field_type == "enum":
        return _cast_enum(stripped, field)

    if field_type == "array":
        return _cast_array(stripped, field)

    # string (constrained or unconstrained) - strip only, no transformation
    return stripped


# ---------------------------------------------------------------------------
# Private type-cast helpers
# ---------------------------------------------------------------------------


def _cast_boolean(raw: str, field: Field) -> bool:
    """Cast 'true'/'false' string to Python bool (case-insensitive).

    Args:
        raw: Stripped raw string value.
        field: Field descriptor for error context.

    Returns:
        Python bool.

    Raises:
        ExtractionError: If the value is not a recognised boolean string.
    """
    lower = raw.lower()
    if lower in ("true", "yes", "1"):
        return True
    if lower in ("false", "no", "0"):
        return False
    raise ExtractionError(
        f"Cannot cast {raw!r} to boolean - expected 'true' or 'false'",
        field=field.path,
    )


def _cast_integer(raw: str, field: Field) -> int:
    """Cast string to Python int.

    Strips whitespace and attempts direct int() conversion. Falls back to
    float → int truncation if the string contains a decimal point.

    Args:
        raw: Stripped raw string value.
        field: Field descriptor for error context.

    Returns:
        Python int.

    Raises:
        ExtractionError: If the value cannot be interpreted as an integer.
    """
    # Exact parse first - keeps precision for very large integers that float() rounds.
    try:
        return int(raw)
    except ValueError:
        pass
    # Then strip formatting (commas/currency/percent/parens), so a figure the model
    # copied verbatim from the document ("2,264,331,000") is not dropped on cast.
    num = coerce_number(raw)
    if num is not None and float(num).is_integer():
        return int(num)
    # Float truncation (LLM may output "30.0" for integer fields).
    try:
        as_float = float(raw)
        if as_float == int(as_float):
            return int(as_float)
    except ValueError:
        pass
    raise ExtractionError(
        f"Cannot cast {raw!r} to integer",
        field=field.path,
    )


def _cast_number(raw: str, field: Field) -> float:
    """Cast string to Python float.

    Args:
        raw: Stripped raw string value.
        field: Field descriptor for error context.

    Returns:
        Python float.

    Raises:
        ExtractionError: If the value cannot be interpreted as a number.
    """
    num = coerce_number(raw)
    if num is not None:
        return num
    try:
        return float(raw)  # fallback for forms coerce_number declines (e.g. "1e3")
    except ValueError:
        raise ExtractionError(
            f"Cannot cast {raw!r} to number",
            field=field.path,
        ) from None


def _cast_enum(raw: str, field: Field) -> str:
    """Validate that a string value is a member of the field's enum set.

    Args:
        raw: Stripped raw string value.
        field: Field descriptor providing ``constraints["enum"]`` list.

    Returns:
        The validated enum string value.

    Raises:
        ExtractionError: If the value is not in the enum set.
    """
    allowed: list[Any] = field.constraints.get("enum", [])
    if not allowed:
        return raw
    # Direct match first (exact string)
    if raw in allowed:
        return raw
    # Case-insensitive fallback for string enums
    raw_lower = raw.lower()
    for option in allowed:
        if isinstance(option, str) and option.lower() == raw_lower:
            return option
    raise ExtractionError(
        f"Value {raw!r} is not a valid enum member. Allowed: {allowed}",
        field=field.path,
    )


def _array_items_are_objects(field: Field) -> bool:
    """Return ``True`` if an array field's items are objects (a JSON list-leaf).

    The flattener stores the item schema under ``constraints["items"]``. An item
    that is an object - directly (``type: object`` / ``properties``) or via a
    ``$ref`` (resolved object) - marks a variable-length array of objects whose
    value is a JSON array, not a bracketed scalar list.
    """
    items = field.constraints.get("items")
    if not isinstance(items, dict):
        return False
    return "$ref" in items or items.get("type") == "object" or "properties" in items


def _cast_object_array(raw: str, field: Field) -> list[Any]:
    """Parse a JSON array of objects, tolerating text around the array.

    Args:
        raw: The stripped SFEP value, expected to be a JSON array of objects.
        field: Field descriptor (for the error path/message).

    Returns:
        A list of dict elements (empty list when the value is empty or ``[]``).

    Raises:
        ExtractionError: If a non-empty value cannot be parsed as a JSON array.
    """
    if not raw or raw in {"[]", "null", "none"}:
        return []
    start = raw.find("[")
    if start == -1:
        raise ExtractionError(
            f"Array field expected a JSON array of objects, got {raw[:40]!r}",
            field=field.path,
        )
    end = raw.rfind("]")
    parsed: Any = None
    if end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
    if not isinstance(parsed, list):
        # Repair miscounted braces and mid-list truncation; salvage as last resort.
        repaired = json_repair.loads(raw[start:])
        parsed = repaired if isinstance(repaired, list) else _salvage_objects(raw[start + 1 :])
    # Repair can wrap a mis-braced group in a stray list; flatten one level.
    props = _item_properties(field)
    rows: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            rows.append(item)
        elif isinstance(item, list):
            rows.extend(x for x in item if isinstance(x, dict))
    if not rows and "{" in raw[start:]:
        # Object rows were emitted but none survived repair: fail so it re-extracts.
        raise ExtractionError(
            f"Array field could not recover any object rows from {raw[:40]!r}",
            field=field.path,
        )
    return [_cast_item(row, props) for row in rows]


def _salvage_objects(inner: str) -> list[dict[str, Any]]:
    """Recover the complete top-level ``{...}`` objects from a truncated JSON array body.

    Walks the string tracking brace depth and string context, parsing each balanced
    object as it closes; stops at the first incomplete one. A model that emitted 8 of
    12 rows before running out of output tokens keeps its 8 instead of losing all.
    """
    objects: list[dict[str, Any]] = []
    depth = 0
    obj_start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(inner):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                try:
                    parsed = json.loads(inner[obj_start : i + 1])
                except json.JSONDecodeError:
                    break
                if isinstance(parsed, dict):
                    objects.append(parsed)
                obj_start = -1
    return objects


def _item_properties(field: Field) -> dict[str, Any]:
    """Return the item object's property schemas for an object array, else empty."""
    items = field.constraints.get("items")
    if isinstance(items, dict):
        props = items.get("properties")
        if isinstance(props, dict):
            return props
    return {}


def _cast_item(item: dict[str, Any], props: dict[str, Any]) -> dict[str, Any]:
    """Coerce each item field to its schema type (numbers out of comma strings)."""
    return {key: _cast_item_value(value, props.get(key)) for key, value in item.items()}


def _cast_item_value(value: Any, prop: Any) -> Any:
    """Coerce one item value to its property schema type; non-strings pass through."""
    if not isinstance(value, str):
        return value
    prop_type = None
    if isinstance(prop, dict):
        prop_type = prop.get("type")
        if isinstance(prop_type, list):
            prop_type = next((t for t in prop_type if t != "null"), None)
    # Typed numbers always coerce; untyped only on a display marker, so codes stay strings.
    if prop_type in ("number", "integer") or (prop_type is None and _has_number_marker(value)):
        number = coerce_number(value)
        if number is not None:
            if prop_type == "integer" or number.is_integer():
                return int(number)
            return number
    return value


def _has_number_marker(value: str) -> bool:
    """True when *value* shows a displayed-number marker (grouping/currency/%/parens)."""
    if not any(ch.isdigit() for ch in value):
        return False
    if any(ch in _CURRENCY_SIGNS for ch in value) or value.rstrip().endswith("%"):
        return True
    if value.startswith("(") and value.endswith(")"):
        return True
    # A grouping separator sitting between two digits (19,715 / 1 234 / 1'234).
    return any(
        value[i] in _GROUP_SIGNS and value[i - 1].isdigit() and value[i + 1].isdigit()
        for i in range(1, len(value) - 1)
    )


def _cast_array(raw: str, field: Field) -> list[Any]:
    """Parse bracket-notation array string into a Python list.

    Supports the format ``[item1, item2, item3]``. Element type is inferred
    from ``field.constraints["items"]["type"]`` when available; defaults to
    string elements.

    Args:
        raw: Stripped raw string value, expected to start/end with brackets.
        field: Field descriptor for element type resolution.

    Returns:
        Python list of typed elements. A single bare value (no brackets, no
        comma) is wrapped as a one-element list ``[value]``.

    Example:
        >>> from nfield.schema._types import Field
        >>> f = Field("tags", "array", {"items": {"type": "string"}}, "", {})
        >>> _cast_array("[alpha, beta, gamma]", f)
        ['alpha', 'beta', 'gamma']
    """
    stripped = raw.strip()

    # An object array carries its whole value as one JSON array; parse it as JSON.
    if _array_items_are_objects(field):
        return _cast_object_array(stripped, field)

    # JSON-first: quoted items containing commas survive the comma split below.
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            item_type = _get_array_item_type(field)
            return [
                _cast_array_element(el, item_type, field) if isinstance(el, str) else el
                for el in parsed
            ]

    # Normalise arrays the model emitted without brackets: a bare comma list
    # becomes a list; a single bare value becomes a one-element list.
    if not (stripped.startswith("[") and stripped.endswith("]")):
        if "," in stripped:
            # Bare comma-separated list - treat as array
            stripped = f"[{stripped}]"
        else:
            # Single bare value (LLM omitted brackets) - wrap as single-element array
            item_type = _get_array_item_type(field)
            element = _cast_array_element(stripped, item_type, field)
            return [element]

    inner = stripped[1:-1].strip()
    if not inner:
        return []

    item_type = _get_array_item_type(field)
    items_raw = _split_array_items(inner)
    return [_cast_array_element(item.strip(), item_type, field) for item in items_raw]


def _get_array_item_type(field: Field) -> str:
    """Extract element type from a field's items constraint.

    Args:
        field: Field descriptor.

    Returns:
        Type string (e.g. ``"string"``, ``"integer"``). Defaults to ``"string"``.
    """
    items_schema = field.constraints.get("items", {})
    if isinstance(items_schema, dict):
        return str(items_schema.get("type", "string"))
    return "string"


def _split_array_items(inner: str) -> list[str]:
    """Split comma-separated array content respecting quoted strings.

    Args:
        inner: String content between the brackets (without ``[`` / ``]``).

    Returns:
        List of raw item strings (not yet typecast).
    """
    items: list[str] = []
    depth = 0
    current: list[str] = []
    in_quote: str | None = None

    for char in inner:
        if in_quote:
            current.append(char)
            if char == in_quote:
                in_quote = None
        elif char in ('"', "'"):
            in_quote = char
            current.append(char)
        elif char == "[":
            depth += 1
            current.append(char)
        elif char == "]":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        items.append("".join(current).strip())

    return [item for item in items if item]


def _cast_array_element(raw: str, item_type: str, field: Field) -> Any:
    """Cast a single array element to the target type.

    Args:
        raw: Raw element string.
        item_type: Expected type string (``"string"``, ``"integer"``, etc.).
        field: Parent field for error context.

    Returns:
        Typed element value.
    """
    # Strip quotes from quoted string elements
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1]

    if item_type == "integer":
        try:
            return int(raw)
        except ValueError:
            return raw  # Degrade gracefully - validation will catch it
    if item_type == "number":
        try:
            return float(raw)
        except ValueError:
            return raw
    if item_type == "boolean":
        lower = raw.lower()
        if lower in ("true", "yes"):
            return True
        if lower in ("false", "no"):
            return False
        return raw
    if raw.upper() == _SFEP_NULL_SENTINEL:
        return None
    return raw
