"""Schema preflight — reject a self-contradictory JSON Schema before any LLM call.

A JSON Schema can be syntactically valid yet *unsatisfiable*: ``minimum > maximum``,
an empty ``enum``, ``minLength > maxLength``. Today such a schema runs the whole
pipeline and every field fails, which reads as a confusing "everything missing". This
catches the contradiction up front and raises a :class:`SchemaError` with the field
path and a fix hint — the "fail before you spend a call" discipline.

Scope is deliberately narrow. Full JSON Schema satisfiability is decidable but up to
2EXPTIME, and the cost is driven by Boolean operators (``not``/``anyOf``/``oneOf``/
``allOf`` merging), ``$ref`` recursion, and ``uniqueItems`` — not by simple bounds. So
this checks **only** per-field, single-keyword-pair contradictions that are
*necessarily* unsatisfiable: each can never reject a valid schema (zero false
rejections). Anything that would need a solver (negation, disjunction, ref merging) is
skipped, not guessed — exactly where published satisfiability tools (jsonsubschema,
arXiv:1911.12651) themselves punt.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, TypeGuard

from formatshield.exceptions import SchemaError

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["preflight_schema"]


def preflight_schema(schema: dict[str, Any]) -> None:
    """Raise :class:`SchemaError` if *schema* contains a provable contradiction.

    Walks the schema's object tree and applies the decidable, zero-false-rejection
    contradiction checks (numeric/string/array/object bounds, empty or unreachable
    enum, ``const`` conflicts, uncompilable ``pattern``). Constructs needing a solver
    (``not``/``anyOf``/``oneOf``/``allOf``/``$ref``) are not inspected. Returns
    ``None`` when no contradiction is found.

    Args:
        schema: A normalised JSON Schema dict (the form Stage 1 consumes).

    Raises:
        SchemaError: On the first provable contradiction, carrying the dot-notation
            field path and a fix hint.

    Example:
        >>> preflight_schema({"type": "object", "properties": {"n": {"type": "integer"}}})
        >>> preflight_schema({"type": "integer", "minimum": 5, "maximum": 3})
        Traceback (most recent call last):
            ...
        formatshield.exceptions.SchemaError: minimum (5) must be <= maximum (3) [field=<root>] hint: ...
    """
    for path, node in _walk(schema, ""):
        _check_node(path, node)


# ---------------------------------------------------------------------------
# Tree walk
# ---------------------------------------------------------------------------


def _walk(node: Any, path: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(path, node)`` for the schema node and every nested object property.

    Recurses ``properties`` (objects) and an array's ``items`` schema. Boolean-combinator
    branches (``anyOf``/``oneOf``/``allOf``/``not``) are not descended into — merging them
    is the solver-grade reasoning this module intentionally avoids.

    Args:
        node: The current schema fragment.
        path: Dot-notation path to this node (``""`` at the root).

    Yields:
        ``(path, node)`` pairs for each dict node to be checked.
    """
    if not isinstance(node, dict):
        return
    yield path or "<root>", node
    properties = node.get("properties")
    if isinstance(properties, dict):
        for key, child in properties.items():
            yield from _walk(child, f"{path}.{key}" if path else key)
    items = node.get("items")
    if isinstance(items, dict):
        yield from _walk(items, f"{path}[]" if path else "[]")


# ---------------------------------------------------------------------------
# Per-node checks (each is a necessary contradiction → zero false rejects)
# ---------------------------------------------------------------------------


def _check_node(path: str, node: dict[str, Any]) -> None:
    """Apply every contradiction check to a single schema node.

    Args:
        path: Dot-notation path to this node, for the error message.
        node: The schema fragment to check.

    Raises:
        SchemaError: On the first contradiction found in this node.
    """
    _check_numeric_bounds(path, node)
    _check_length_bounds(path, node, "minLength", "maxLength", "string length")
    _check_length_bounds(path, node, "minItems", "maxItems", "array length")
    _check_length_bounds(path, node, "minProperties", "maxProperties", "property count")
    _check_pattern_compiles(path, node)
    _check_enum(path, node)
    _check_const(path, node)
    _check_required_present(path, node)
    _check_multiple_of_range(path, node)


def _check_numeric_bounds(path: str, node: dict[str, Any]) -> None:
    """Reject conflicting numeric bounds (inclusive and exclusive)."""
    minimum = _number(node.get("minimum"))
    maximum = _number(node.get("maximum"))
    excl_min = _number(node.get("exclusiveMinimum"))
    excl_max = _number(node.get("exclusiveMaximum"))

    if minimum is not None and maximum is not None and minimum > maximum:
        _fail(
            path,
            f"minimum ({_fmt(minimum)}) must be <= maximum ({_fmt(maximum)})",
            node,
            "minimum",
            "set minimum <= maximum",
        )
    if excl_min is not None and excl_max is not None and excl_min >= excl_max:
        _fail(
            path,
            f"exclusiveMinimum ({_fmt(excl_min)}) must be < exclusiveMaximum ({_fmt(excl_max)})",
            node,
            "exclusiveMinimum",
            "widen the exclusive range",
        )
    if excl_min is not None and maximum is not None and excl_min >= maximum:
        _fail(
            path,
            f"exclusiveMinimum ({_fmt(excl_min)}) leaves no value <= maximum ({_fmt(maximum)})",
            node,
            "exclusiveMinimum",
            "lower exclusiveMinimum or raise maximum",
        )
    if minimum is not None and excl_max is not None and minimum >= excl_max:
        _fail(
            path,
            f"minimum ({_fmt(minimum)}) leaves no value < exclusiveMaximum ({_fmt(excl_max)})",
            node,
            "minimum",
            "raise exclusiveMaximum or lower minimum",
        )


def _check_length_bounds(
    path: str, node: dict[str, Any], lo_key: str, hi_key: str, what: str
) -> None:
    """Reject ``min* > max*`` for a length/count keyword pair."""
    lo = _number(node.get(lo_key))
    hi = _number(node.get(hi_key))
    if lo is not None and hi is not None and lo > hi:
        _fail(
            path,
            f"{lo_key} ({_fmt(lo)}) must be <= {hi_key} ({_fmt(hi)}) for {what}",
            node,
            lo_key,
            f"set {lo_key} <= {hi_key}",
        )


def _check_pattern_compiles(path: str, node: dict[str, Any]) -> None:
    """Reject a ``pattern`` that is not a compilable regular expression."""
    pattern = node.get("pattern")
    if isinstance(pattern, str):
        try:
            re.compile(pattern)
        except re.error as exc:
            _fail(
                path,
                f"pattern {pattern!r} is not a valid regular expression: {exc}",
                node,
                "pattern",
                "fix the regular expression",
            )


def _check_enum(path: str, node: dict[str, Any]) -> None:
    """Reject an empty enum, or an enum whose every member fails a sibling constraint."""
    enum = node.get("enum")
    if not isinstance(enum, list):
        return
    if len(enum) == 0:
        _fail(
            path,
            "enum is empty, so no value can satisfy it",
            node,
            "enum",
            "add at least one allowed value",
        )
    declared = _declared_types(node)
    if declared and all(not _matches_type(member, declared) for member in enum):
        _fail(
            path,
            f"no enum member matches the declared type(s) {sorted(declared)}",
            node,
            "enum",
            "align the enum values with the declared type",
        )
    pattern = node.get("pattern")
    if isinstance(pattern, str):
        try:
            compiled = re.compile(pattern)
        except re.error:
            return  # pattern validity is reported by _check_pattern_compiles
        str_members = [m for m in enum if isinstance(m, str)]
        if str_members and all(compiled.search(m) is None for m in str_members):
            _fail(
                path,
                f"no enum member matches pattern /{pattern}/",
                node,
                "enum",
                "make an enum value match the pattern, or drop the pattern",
            )


def _check_const(path: str, node: dict[str, Any]) -> None:
    """Reject a ``const`` that contradicts a sibling type, enum, or numeric bound."""
    if "const" not in node:
        return
    value = node["const"]
    declared = _declared_types(node)
    if declared and not _matches_type(value, declared):
        _fail(
            path,
            f"const {value!r} does not match the declared type(s) {sorted(declared)}",
            node,
            "const",
            "set const to a value of the declared type",
        )
    enum = node.get("enum")
    if isinstance(enum, list) and value not in enum:
        _fail(
            path,
            f"const {value!r} is not a member of the enum",
            node,
            "const",
            "make const one of the enum values",
        )
    number = _number(value)
    if number is not None:
        minimum = _number(node.get("minimum"))
        maximum = _number(node.get("maximum"))
        if minimum is not None and number < minimum:
            _fail(
                path,
                f"const {_fmt(number)} is below minimum ({_fmt(minimum)})",
                node,
                "const",
                "raise const or lower minimum",
            )
        if maximum is not None and number > maximum:
            _fail(
                path,
                f"const {_fmt(number)} is above maximum ({_fmt(maximum)})",
                node,
                "const",
                "lower const or raise maximum",
            )


def _check_required_present(path: str, node: dict[str, Any]) -> None:
    """Reject a required key that is closed out by ``additionalProperties: false``.

    Only fires when the key is absent from ``properties`` AND ``additionalProperties``
    is ``false`` AND there is no ``patternProperties`` that could admit it — the only
    case where the requirement is provably unsatisfiable.
    """
    required = node.get("required")
    if not isinstance(required, list) or node.get("additionalProperties") is not False:
        return
    if "patternProperties" in node:
        return
    properties = node.get("properties")
    known = set(properties) if isinstance(properties, dict) else set()
    for key in required:
        if isinstance(key, str) and key not in known:
            _fail(
                path,
                f"required property {key!r} is absent from properties while "
                "additionalProperties is false, so it can never be present",
                node,
                "required",
                "add the property to 'properties' or allow additionalProperties",
            )


def _check_multiple_of_range(path: str, node: dict[str, Any]) -> None:
    """Reject an integer ``multipleOf`` whose multiples all fall outside [minimum, maximum].

    Restricted to integer multipleOf with integer bounds — float multipleOf invites
    precision error, so it is skipped to preserve the zero-false-rejection guarantee.
    """
    multiple = node.get("multipleOf")
    minimum = node.get("minimum")
    maximum = node.get("maximum")
    if not (_is_int(multiple) and _is_int(minimum) and _is_int(maximum)):
        return
    if multiple <= 0:
        return
    # Smallest multiple of `multiple` that is >= minimum; if it exceeds maximum, the
    # range admits no multiple at all.
    lowest = math.ceil(minimum / multiple) * multiple
    if lowest > maximum:
        _fail(
            path,
            f"no multiple of {multiple} lies in [{minimum}, {maximum}]",
            node,
            "multipleOf",
            "widen the range or change multipleOf",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _declared_types(node: dict[str, Any]) -> set[str]:
    """Return the node's declared ``type``(s) as a set, or empty if untyped."""
    declared = node.get("type")
    if isinstance(declared, str):
        return {declared}
    if isinstance(declared, list):
        return {t for t in declared if isinstance(t, str)}
    return set()


def _matches_type(value: Any, types: set[str]) -> bool:
    """Return whether *value* could satisfy at least one of the JSON Schema *types*."""
    for t in types:
        if t == "string" and isinstance(value, str):
            return True
        if t == "boolean" and isinstance(value, bool):
            return True
        if t == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if t == "array" and isinstance(value, list):
            return True
        if t == "object" and isinstance(value, dict):
            return True
        if t == "null" and value is None:
            return True
    return False


def _number(value: Any) -> float | None:
    """Return *value* as a float if it is a real number (not bool), else ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_int(value: Any) -> TypeGuard[int]:
    """Return whether *value* is a plain int (not bool)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _fmt(value: float) -> str:
    """Render a number without a trailing ``.0`` for whole values."""
    return str(int(value)) if value == int(value) else str(value)


def _fail(path: str, message: str, node: dict[str, Any], keyword: str, hint: str) -> None:
    """Raise a :class:`SchemaError` for a contradiction at *path*.

    Args:
        path: Dot-notation field path.
        message: What is contradictory (with the offending values inline).
        node: The schema node (unused beyond context; kept for future detail).
        keyword: The schema keyword at fault (for the hint context).
        hint: A concrete fix suggestion.

    Raises:
        SchemaError: Always.
    """
    raise SchemaError(message, field=path, hint=hint)
