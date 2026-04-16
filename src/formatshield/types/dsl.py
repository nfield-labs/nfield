"""FormatShield type DSL — composable constraint types for structured generation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import EnumMeta
from typing import Any, Union, get_args, get_origin

_logger = logging.getLogger(__name__)

# ── Base Term ─────────────────────────────────────────────────────────────────


class Term:
    """Base class for all output type constraints."""

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return f"{type(self).__name__}()"

    def __or__(self, other: Term | str) -> Alternatives:
        """Create an Alternatives constraint from self and other."""
        other = String(str(other)) if isinstance(other, str) else other
        return Alternatives([self, other])

    def __ror__(self, other: Term | str) -> Alternatives:
        """Create an Alternatives constraint from other and self."""
        other = String(str(other)) if isinstance(other, str) else other
        return Alternatives([other, self])

    def __add__(self, other: Term | str) -> Sequence:
        """Create a Sequence constraint from self followed by other."""
        other = String(str(other)) if isinstance(other, str) else other
        return Sequence([self, other])

    def __radd__(self, other: Term | str) -> Sequence:
        """Create a Sequence constraint from other followed by self."""
        other = String(str(other)) if isinstance(other, str) else other
        return Sequence([other, self])

    def optional(self) -> Optional:
        """Wrap self in an Optional constraint."""
        return Optional(self)

    def exactly(self, n: int) -> QuantifyExact:
        """Require exactly n repetitions of self."""
        return QuantifyExact(n, self)

    def at_least(self, n: int) -> QuantifyMinimum:
        """Require at least n repetitions of self."""
        return QuantifyMinimum(n, self)

    def at_most(self, n: int) -> QuantifyMaximum:
        """Require at most n repetitions of self."""
        return QuantifyMaximum(n, self)

    def between(self, min_count: int, max_count: int) -> QuantifyBetween:
        """Require between min_count and max_count repetitions of self."""
        return QuantifyBetween(min_count, max_count, self)

    def one_or_more(self) -> KleenePlus:
        """Require one or more repetitions of self."""
        return KleenePlus(self)

    def zero_or_more(self) -> KleeneStar:
        """Require zero or more repetitions of self."""
        return KleeneStar(self)

    def matches(self, value: str) -> bool:
        """Return True if value matches this constraint.

        Args:
            value: The string to test.

        Returns:
            True if *value* satisfies the constraint, False otherwise.
        """
        try:
            pattern = to_regex(self)
            return bool(re.fullmatch(pattern, value))
        except Exception:
            return False

    def validate(self, value: str) -> str:
        """Validate value; raise ValueError if not matching.

        Args:
            value: The string to validate.

        Returns:
            The original *value* when it matches.

        Raises:
            ValueError: If *value* does not satisfy the constraint.
        """
        if not self.matches(value):
            raise ValueError(f"Value {value!r} does not match constraint {self!r}")
        return value


@dataclass
class String(Term):
    """Exact literal string."""

    value: str

    def __repr__(self) -> str:
        return f"String({self.value!r})"


@dataclass
class Regex(Term):
    """Regex pattern constraint."""

    pattern: str

    def __repr__(self) -> str:
        return f"Regex({self.pattern!r})"


@dataclass
class CFG(Term):
    """Context-free grammar (Lark EBNF) constraint."""

    definition: str

    def __repr__(self) -> str:
        return "CFG(...)"

    @classmethod
    def from_file(cls, path: str) -> CFG:
        """Load grammar from a .lark file.

        Args:
            path: Path to the Lark grammar file.

        Returns:
            A CFG constraint with the file's content.
        """
        with open(path) as f:
            return cls(f.read())


@dataclass
class JsonSchema(Term):
    """JSON schema constraint."""

    schema: str
    whitespace_pattern: str | None = None
    ensure_ascii: bool = True

    def __init__(
        self,
        schema: dict | str | type,
        whitespace_pattern: str | None = None,
        ensure_ascii: bool = True,
    ) -> None:
        import json

        if isinstance(schema, str):
            self.schema = schema
        elif isinstance(schema, dict):
            self.schema = json.dumps(schema)
        else:
            # Pydantic model or dataclass
            self.schema = _schema_from_type(schema)
        self.whitespace_pattern = whitespace_pattern
        self.ensure_ascii = ensure_ascii

    def __repr__(self) -> str:
        return "JsonSchema(...)"

    @classmethod
    def from_file(cls, path: str) -> JsonSchema:
        """Load JSON schema from a file.

        Args:
            path: Path to the JSON schema file.

        Returns:
            A JsonSchema constraint with the file's content.
        """
        with open(path) as f:
            return cls(f.read())

    @staticmethod
    def is_json_schema(obj: Any) -> bool:
        """Return True if obj looks like a JSON schema dict.

        Args:
            obj: Object to inspect.

        Returns:
            True when *obj* is a dict with ``type``, ``properties``,
            ``$schema``, or ``anyOf`` keys.
        """
        return isinstance(obj, dict) and (
            "type" in obj or "properties" in obj or "$schema" in obj or "anyOf" in obj
        )


@dataclass
class Choice(Term):
    """Constrain output to one of a fixed set of string choices."""

    items: list[Any]

    def __repr__(self) -> str:
        return f"Choice({self.items!r})"


@dataclass
class KleeneStar(Term):
    """Zero or more repetitions."""

    term: Term


@dataclass
class KleenePlus(Term):
    """One or more repetitions."""

    term: Term


@dataclass
class Optional(Term):
    """Zero or one occurrence."""

    term: Term


@dataclass
class Alternatives(Term):
    """Union of multiple terms (any one matches)."""

    terms: list[Term]


@dataclass
class Sequence(Term):
    """Concatenation of multiple terms."""

    terms: list[Term]


@dataclass
class QuantifyExact(Term):
    """Exactly N repetitions."""

    count: int
    term: Term


@dataclass
class QuantifyMinimum(Term):
    """At least N repetitions."""

    count: int
    term: Term


@dataclass
class QuantifyMaximum(Term):
    """At most N repetitions."""

    count: int
    term: Term


@dataclass
class QuantifyBetween(Term):
    """Between min and max repetitions."""

    min_count: int
    max_count: int
    term: Term


# ── DSL factory functions ─────────────────────────────────────────────────────


def regex(pattern: str) -> Regex:
    """Create a Regex constraint.

    Args:
        pattern: A regular expression string.

    Returns:
        A :class:`Regex` term.
    """
    return Regex(pattern)


def cfg(definition: str) -> CFG:
    """Create a CFG constraint.

    Args:
        definition: A Lark EBNF grammar string.

    Returns:
        A :class:`CFG` term.
    """
    return CFG(definition)


def json_schema(schema: dict | str | type) -> JsonSchema:
    """Create a JsonSchema constraint.

    Args:
        schema: A JSON schema dict, JSON string, or Pydantic/dataclass type.

    Returns:
        A :class:`JsonSchema` term.
    """
    return JsonSchema(schema)


def either(*terms: Term | str) -> Alternatives:
    """Create an Alternatives constraint from multiple terms.

    Args:
        *terms: Two or more terms (or strings) to combine as alternatives.

    Returns:
        An :class:`Alternatives` term.
    """
    converted = [String(t) if isinstance(t, str) else t for t in terms]
    return Alternatives(converted)


def optional(term: Term | str) -> Optional:
    """Create an Optional constraint.

    Args:
        term: The term to make optional.

    Returns:
        An :class:`Optional` term.
    """
    return Optional(String(term) if isinstance(term, str) else term)


def exactly(count: int, term: Term | str) -> QuantifyExact:
    """Exactly N repetitions.

    Args:
        count: Number of required repetitions.
        term: The term to repeat.

    Returns:
        A :class:`QuantifyExact` term.
    """
    return QuantifyExact(count, String(term) if isinstance(term, str) else term)


def at_least(count: int, term: Term | str) -> QuantifyMinimum:
    """At least N repetitions.

    Args:
        count: Minimum number of repetitions.
        term: The term to repeat.

    Returns:
        A :class:`QuantifyMinimum` term.
    """
    return QuantifyMinimum(count, String(term) if isinstance(term, str) else term)


def at_most(count: int, term: Term | str) -> QuantifyMaximum:
    """At most N repetitions.

    Args:
        count: Maximum number of repetitions.
        term: The term to repeat.

    Returns:
        A :class:`QuantifyMaximum` term.
    """
    return QuantifyMaximum(count, String(term) if isinstance(term, str) else term)


def between(min_count: int, max_count: int, term: Term | str) -> QuantifyBetween:
    """Between min and max repetitions.

    Args:
        min_count: Minimum number of repetitions.
        max_count: Maximum number of repetitions.
        term: The term to repeat.

    Returns:
        A :class:`QuantifyBetween` term.
    """
    return QuantifyBetween(min_count, max_count, String(term) if isinstance(term, str) else term)


def zero_or_more(term: Term | str) -> KleeneStar:
    """Zero or more repetitions.

    Args:
        term: The term to repeat.

    Returns:
        A :class:`KleeneStar` term.
    """
    return KleeneStar(String(term) if isinstance(term, str) else term)


def one_or_more(term: Term | str) -> KleenePlus:
    """One or more repetitions.

    Args:
        term: The term to repeat.

    Returns:
        A :class:`KleenePlus` term.
    """
    return KleenePlus(String(term) if isinstance(term, str) else term)


# ── to_regex ──────────────────────────────────────────────────────────────────


def to_regex(term: Term) -> str:
    """Compile a Term tree into a regex string.

    Args:
        term: The constraint term to compile.

    Returns:
        A regular expression string matching the constraint.

    Raises:
        TypeError: If *term* is a :class:`JsonSchema` or :class:`CFG`, which
            cannot be expressed as a plain regex.
        TypeError: If *term* has an unknown type.
    """
    if isinstance(term, String):
        return re.escape(term.value)
    elif isinstance(term, Regex):
        return term.pattern
    elif isinstance(term, Choice):
        return "(" + "|".join(re.escape(str(item)) for item in term.items) + ")"
    elif isinstance(term, Alternatives):
        return "(" + "|".join(to_regex(t) for t in term.terms) + ")"
    elif isinstance(term, Sequence):
        return "".join(to_regex(t) for t in term.terms)
    elif isinstance(term, Optional):
        return f"({to_regex(term.term)})?"
    elif isinstance(term, KleeneStar):
        return f"({to_regex(term.term)})*"
    elif isinstance(term, KleenePlus):
        return f"({to_regex(term.term)})+"
    elif isinstance(term, QuantifyExact):
        return f"({to_regex(term.term)}){{{term.count}}}"
    elif isinstance(term, QuantifyMinimum):
        return f"({to_regex(term.term)}){{{term.count},}}"
    elif isinstance(term, QuantifyMaximum):
        return f"({to_regex(term.term)}){{0,{term.count}}}"
    elif isinstance(term, QuantifyBetween):
        return f"({to_regex(term.term)}){{{term.min_count},{term.max_count}}}"
    elif isinstance(term, JsonSchema):
        raise TypeError(
            "JsonSchema cannot be compiled to regex directly. Use a JSON-schema-aware backend."
        )
    elif isinstance(term, CFG):
        raise TypeError("CFG cannot be compiled to regex directly. Use a grammar backend.")
    else:
        raise TypeError(f"Unknown Term type: {type(term)}")


# ── python_types_to_terms ─────────────────────────────────────────────────────

_MAX_RECURSION_DEPTH = 10


def python_types_to_terms(ptype: Any, recursion_depth: int = 0) -> Term:
    """Convert an arbitrary Python type to a constraint Term.

    Handles: ``int``, ``float``, ``bool``, ``str``, ``dict``, ``list``,
    ``tuple``, Pydantic ``BaseModel``, ``dataclass``, ``TypedDict``,
    ``Enum``, ``Literal``, ``Union``, ``Optional``.

    Args:
        ptype: The Python type or annotation to convert.  May also be an
            existing :class:`Term` instance, which is returned as-is.
        recursion_depth: Internal recursion counter; do not pass manually.

    Returns:
        A :class:`Term` representing the structural constraint for *ptype*.

    Raises:
        RecursionError: If the recursion depth exceeds
            :data:`_MAX_RECURSION_DEPTH` (circular type references).
        TypeError: If *ptype* cannot be mapped to any known constraint.

    Example:
        >>> from enum import Enum
        >>> class Color(str, Enum):
        ...     red = "red"
        ...     blue = "blue"
        >>> term = python_types_to_terms(Color)
        >>> isinstance(term, Choice)
        True
    """
    import dataclasses
    import json
    import typing

    if recursion_depth > _MAX_RECURSION_DEPTH:
        raise RecursionError(
            f"python_types_to_terms() recursion depth exceeded {_MAX_RECURSION_DEPTH}. "
            "Check for circular type references."
        )

    # Already a Term — pass through unchanged.
    if isinstance(ptype, Term):
        return ptype

    # Primitive types → regex
    if ptype is int:
        return Regex(r"[+-]?(0|[1-9][0-9]*)")
    if ptype is float:
        return Regex(r"[+-]?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-][0-9]+)?")
    if ptype is bool:
        return Choice(["true", "false"])
    if ptype is str:
        return Regex(r'"[^"]*"')

    origin = get_origin(ptype)
    args = get_args(ptype)

    # Literal["a", "b"] → Choice
    if origin is typing.Literal:
        return Choice(list(args))

    # Union / Optional → Alternatives (handles both typing.Union and 3.10+ X | Y syntax)
    _is_union = origin is Union
    if not _is_union and hasattr(typing, "get_type_hints"):
        import types as _types

        _is_union = isinstance(ptype, _types.UnionType) if hasattr(_types, "UnionType") else False
    if _is_union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            inner = python_types_to_terms(non_none[0], recursion_depth + 1)
            return Optional(inner)
        terms = [python_types_to_terms(a, recursion_depth + 1) for a in non_none]
        return Alternatives(terms)

    # Enum → Choice of values
    if isinstance(ptype, EnumMeta):
        return Choice([member.value for member in ptype])  # type: ignore[attr-defined]

    # Pydantic BaseModel
    try:
        from pydantic import BaseModel

        if isinstance(ptype, type) and issubclass(ptype, BaseModel):
            schema = ptype.model_json_schema()
            return JsonSchema(json.dumps(schema))
    except ImportError:
        pass

    # dataclass
    if dataclasses.is_dataclass(ptype) and isinstance(ptype, type):
        try:
            from pydantic import TypeAdapter

            schema = TypeAdapter(ptype).json_schema()
            return JsonSchema(json.dumps(schema))
        except Exception as exc:
            _logger.debug("Pydantic TypeAdapter failed for dataclass %r: %s", ptype, exc)
        # Fallback: build schema manually
        fields = {f.name: {"type": "string"} for f in dataclasses.fields(ptype)}
        schema_dict: dict[str, Any] = {"type": "object", "properties": fields}
        return JsonSchema(json.dumps(schema_dict))

    # TypedDict
    if isinstance(ptype, type) and issubclass(ptype, dict) and hasattr(ptype, "__annotations__"):
        try:
            from pydantic import TypeAdapter

            schema = TypeAdapter(ptype).json_schema()
            return JsonSchema(json.dumps(schema))
        except Exception:
            fields = {k: {"type": "string"} for k in ptype.__annotations__}
            schema_dict = {"type": "object", "properties": fields}
            return JsonSchema(json.dumps(schema_dict))

    # dict → JSON object (no schema)
    if ptype is dict or origin is dict:
        return JsonSchema(json.dumps({"type": "object"}))

    # list[T]
    if origin is list:
        if args:
            inner = python_types_to_terms(args[0], recursion_depth + 1)
            if isinstance(inner, JsonSchema):
                import json as _json

                inner_schema = _json.loads(inner.schema)
                array_schema: dict[str, Any] = {"type": "array", "items": inner_schema}
                return JsonSchema(json.dumps(array_schema))
        return JsonSchema(json.dumps({"type": "array"}))

    # Plain dict as JSON schema
    if isinstance(ptype, dict):
        return JsonSchema(json.dumps(ptype))

    raise TypeError(
        f"Cannot convert {ptype!r} to a FormatShield constraint Term. "
        "Supported types: int, float, bool, str, Enum, Literal, Union, Optional, "
        "Pydantic BaseModel, dataclass, TypedDict, list[T], dict."
    )


# ── Internal helpers ──────────────────────────────────────────────────────────


def _schema_from_type(ptype: type) -> str:
    """Extract JSON schema string from a Pydantic model or dataclass.

    Args:
        ptype: A Pydantic BaseModel subclass or a dataclass type.

    Returns:
        A JSON-encoded schema string.
    """
    import json

    try:
        from pydantic import BaseModel

        if issubclass(ptype, BaseModel):
            return json.dumps(ptype.model_json_schema())
    except (ImportError, TypeError):
        pass
    import dataclasses

    if dataclasses.is_dataclass(ptype):
        try:
            from pydantic import TypeAdapter

            return json.dumps(TypeAdapter(ptype).json_schema())
        except Exception as exc:
            _logger.debug("Pydantic TypeAdapter failed for type %r: %s", ptype, exc)
    return json.dumps({"type": "object"})
