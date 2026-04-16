"""
Partial[T] — progressive streaming response type.

When a model streams tokens, ``Partial[T]`` allows callers to receive partially
populated instances of ``T`` as tokens arrive, rather than waiting for the
full response. Each streamed object is valid Python (missing fields are ``None``),
so the caller can progressively update a UI or pipeline stage.

Usage::

    from formatshield.dsl import Partial
    from pydantic import BaseModel

    class Analysis(BaseModel):
        summary: str
        sentiment: str
        confidence: float

    # Build the schema for partial streaming:
    schema = Partial.build_schema(Analysis)

    # Parse a partial JSON fragment:
    partial = Partial.parse_partial('{"summary": "The product is", "sentiment": null}', Analysis)
    # partial.summary == "The product is"
    # partial.sentiment is None (field not yet filled)
    # partial.confidence is None (field not yet filled)
"""

from __future__ import annotations

import json
from typing import Any, Generic, TypeVar

T = TypeVar("T")


def _make_all_optional(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *schema* with all required fields removed.

    Used to generate a schema that accepts partially populated objects — any
    missing field is treated as ``null`` rather than a validation error.

    Args:
        schema: A JSON schema dict.

    Returns:
        Modified schema with ``required`` removed and all properties marked
        as nullable via ``anyOf: [{...}, {"type": "null"}]``.
    """
    result = dict(schema)
    result.pop("required", None)

    if "properties" in result:
        nullable_props: dict[str, Any] = {}
        for prop_name, prop_schema in result["properties"].items():
            nullable_props[prop_name] = {
                "anyOf": [prop_schema, {"type": "null"}],
                "default": None,
            }
        result["properties"] = nullable_props

    return result


class Partial(Generic[T]):
    """Type-level marker for progressive streaming outputs.

    Use ``Partial[T]`` as the ``output_type`` to signal that the caller
    wants a stream of partially populated ``T`` instances rather than a
    single complete result.

    ``Partial`` itself is not instantiated — it carries typing metadata
    and provides schema/parsing helpers.

    Example::

        result = await shield.generate(prompt, output_type=Partial[Analysis])
        # result.parsed is an Analysis instance with some fields None
    """

    _wrapped_type: type[Any]

    def __class_getitem__(cls, item: type[T]) -> type[Partial[T]]:  # type: ignore[override]
        """Support ``Partial[T]`` subscript syntax."""
        new_cls: type[Partial[T]] = type(  # type: ignore[assignment]
            f"Partial[{getattr(item, '__name__', str(item))}]",
            (cls,),
            {"_wrapped_type": item},
        )
        return new_cls

    @classmethod
    def build_schema(cls, model_type: type[Any]) -> dict[str, Any]:
        """Build a permissive JSON schema for partial streaming.

        All fields are made nullable and ``required`` is removed so that
        incomplete model outputs validate as partial objects.

        Args:
            model_type: The target Pydantic model or type.

        Returns:
            JSON schema dict with all fields optional and nullable.
        """
        if hasattr(model_type, "model_json_schema"):
            base_schema: dict[str, Any] = model_type.model_json_schema()
        else:
            base_schema = {"type": "object"}

        return _make_all_optional(base_schema)

    @classmethod
    def parse_partial(cls, raw: str, model_type: type[Any]) -> Any:
        """Parse a (possibly incomplete) JSON string into a partial model.

        Fills missing fields with ``None`` rather than raising. Best-effort:
        if the JSON is completely invalid, returns ``None``.

        Args:
            raw: Raw string output from the backend (may be truncated).
            model_type: The target Pydantic model class.

        Returns:
            An instance of ``model_type`` with missing fields as ``None``,
            or ``None`` if ``raw`` cannot be parsed at all.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to recover from truncated JSON by appending closing braces
            recovered = _try_recover_json(raw)
            if recovered is None:
                return None
            data = recovered

        if not isinstance(data, dict):
            return None

        # Fill missing fields with None
        if hasattr(model_type, "model_fields"):
            for field_name in model_type.model_fields:
                if field_name not in data:
                    data[field_name] = None

        try:
            if hasattr(model_type, "model_validate"):
                return model_type.model_validate(data)
            return model_type(**data)
        except Exception:
            # Return a best-effort dict if model validation fails
            return data

    @classmethod
    def get_wrapped_type(cls) -> type[Any]:
        """Return the wrapped type ``T`` from ``Partial[T]``.

        Returns:
            The type argument passed to ``Partial[T]``.

        Raises:
            AttributeError: If called on the base ``Partial`` class (not subscripted).
        """
        return cls._wrapped_type  # type: ignore[attr-defined]


def _try_recover_json(raw: str) -> Any | None:
    """Attempt to complete a truncated JSON string by closing open braces/brackets.

    Handles the common case of a streaming response that was cut off mid-token.

    Args:
        raw: Potentially truncated JSON string.

    Returns:
        Parsed Python object if recovery succeeds, ``None`` otherwise.
    """
    # Count unclosed braces/brackets
    stack: list[str] = []
    in_string = False
    escape_next = False
    closing = {"{": "}", "[": "]"}

    for char in raw:
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in ("{", "["):
            stack.append(char)
        elif char in ("}", "]") and stack:
            stack.pop()

    # Close the unclosed structures in reverse order
    suffix = "".join(closing[c] for c in reversed(stack))
    candidate = raw.rstrip(",") + suffix

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None
