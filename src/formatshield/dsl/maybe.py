"""
Maybe[T] — optional result type for uncertain model outputs.

When a model is uncertain about an extraction or classification, ``Maybe[T]``
wraps the response in a ``MaybeResult`` that carries both the (optional) parsed
value and an error description. This prevents ``ValidationError`` from propagating
when the model correctly identifies that it cannot answer.

Usage::

    from formatshield.dsl import Maybe
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str
        age: int

    # Build the JSON schema that FormatShield sends to the backend:
    schema = Maybe.build_schema(Person)
    # → {"type": "object", "properties": {"result": {...}, "error": {...}, ...}}

    # Parse a raw model response:
    maybe = MaybeResult.from_raw('{"result": {"name": "Alice", "age": 30}, "error": false}', Person)
    assert maybe.result is not None
    assert maybe.result.name == "Alice"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class MaybeResult(Generic[T]):
    """Wraps a model response that may or may not contain a valid result.

    Attributes:
        result: The parsed value of type ``T``, or ``None`` if the model
            could not produce a valid answer.
        error: ``True`` if the model flagged uncertainty or failure.
        error_message: Human-readable explanation when ``error`` is ``True``.
    """

    result: T | None
    error: bool
    error_message: str | None = None

    @classmethod
    def from_raw(cls, raw: str, model_type: type[T]) -> MaybeResult[T]:
        """Parse a raw JSON string into a ``MaybeResult``.

        Expects the model to have returned JSON matching the schema produced
        by :meth:`Maybe.build_schema`. Falls back gracefully if the JSON is
        malformed.

        Args:
            raw: Raw string output from the backend.
            model_type: The Pydantic BaseModel class or plain Python type
                to instantiate from the ``result`` field.

        Returns:
            :class:`MaybeResult` with ``result`` populated if successful,
            or ``error=True`` if parsing or validation failed.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return cls(result=None, error=True, error_message=f"Invalid JSON: {raw!r}")

        if not isinstance(data, dict):
            return cls(result=None, error=True, error_message="Response is not a JSON object")

        has_error = bool(data.get("error", False))
        error_msg = data.get("error_message") or data.get("error_reason")

        if has_error or data.get("result") is None:
            return cls(result=None, error=True, error_message=error_msg)

        # Attempt to construct the target type
        raw_result = data["result"]
        try:
            if hasattr(model_type, "model_validate"):
                # Pydantic BaseModel
                parsed: T = model_type.model_validate(raw_result)  # type: ignore[attr-defined]
            elif callable(model_type):
                parsed = model_type(raw_result)  # type: ignore[call-arg]
            else:
                parsed = raw_result  # type: ignore[assignment]
        except Exception as exc:
            return cls(
                result=None,
                error=True,
                error_message=f"Validation failed: {exc}",
            )

        return cls(result=parsed, error=False, error_message=None)

    def unwrap(self) -> T:
        """Return the result or raise ``ValueError`` if it is absent.

        Raises:
            ValueError: If ``result`` is ``None`` (i.e., ``error`` is ``True``).
        """
        if self.result is None:
            raise ValueError(f"MaybeResult has no value. error_message={self.error_message!r}")
        return self.result

    def unwrap_or(self, default: T) -> T:
        """Return the result, or *default* if it is absent.

        Args:
            default: Fallback value returned when ``result`` is ``None``.
        """
        return self.result if self.result is not None else default


class Maybe(Generic[T]):
    """Type-level marker for optional model outputs.

    Use ``Maybe[T]`` as the ``output_type`` argument to trigger the
    ``MaybeResult`` wrapping behaviour in ``FormatShield.generate()``.

    ``Maybe`` itself is not instantiated — it only carries typing metadata
    and provides :meth:`build_schema` for schema construction.

    Example::

        result = await shield.generate(prompt, output_type=Maybe[Person])
        # result.parsed is a MaybeResult[Person]
    """

    _wrapped_type: type[Any]

    def __class_getitem__(cls, item: type[T]) -> type[Maybe[T]]:  # type: ignore[override]
        """Support ``Maybe[T]`` subscript syntax."""
        new_cls: type[Maybe[T]] = type(  # type: ignore[assignment]
            f"Maybe[{getattr(item, '__name__', str(item))}]",
            (cls,),
            {"_wrapped_type": item},
        )
        return new_cls

    @classmethod
    def build_schema(cls, model_type: type[Any]) -> dict[str, Any]:
        """Build the JSON schema for ``Maybe[model_type]``.

        The schema instructs the model to return a JSON object with:

        - ``result``: The structured data (nullable).
        - ``error``: Boolean flag — ``true`` if the model cannot answer.
        - ``error_message``: Optional explanation when ``error`` is ``true``.

        Args:
            model_type: The target Pydantic model or type.

        Returns:
            JSON schema dict.
        """
        # Extract the inner schema for the result field
        if hasattr(model_type, "model_json_schema"):
            inner_schema: dict[str, Any] = model_type.model_json_schema()
        else:
            inner_schema = {"type": "object"}

        return {
            "type": "object",
            "properties": {
                "result": {
                    "oneOf": [inner_schema, {"type": "null"}],
                    "description": "The extracted result, or null if the model cannot answer.",
                },
                "error": {
                    "type": "boolean",
                    "description": "Set to true if the model cannot produce a valid result.",
                },
                "error_message": {
                    "type": ["string", "null"],
                    "description": "Explanation of why the result is absent (when error=true).",
                },
            },
            "required": ["error"],
        }

    @classmethod
    def get_wrapped_type(cls) -> type[Any]:
        """Return the wrapped type ``T`` from ``Maybe[T]``.

        Returns:
            The type argument passed to ``Maybe[T]``.

        Raises:
            AttributeError: If called on the base ``Maybe`` class (not subscripted).
        """
        return cls._wrapped_type  # type: ignore[attr-defined]
