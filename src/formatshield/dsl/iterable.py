"""
IterableModel[T] — stream a sequence of structured model outputs.

When a model needs to produce a list of items (e.g., extract all entities,
enumerate steps), ``IterableModel[T]`` requests the model emit items one by
one rather than as a single large array. This reduces latency to first result
and enables progressive display.

Usage::

    from formatshield.dsl import IterableModel
    from pydantic import BaseModel

    class Entity(BaseModel):
        name: str
        type: str

    # Build the streaming schema:
    schema = IterableModel.build_schema(Entity)
    # → {"type": "array", "items": {...Entity schema...}}

    # Parse items from a completed response:
    items = IterableModel.parse_items('[ {"name": "Alice", "type": "PERSON"} ]', Entity)
    # → [Entity(name="Alice", type="PERSON")]
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class IterableModel(Generic[T]):
    """Type-level marker for streaming sequences of structured outputs.

    Use ``IterableModel[T]`` as the ``output_type`` to signal that the model
    should return a JSON array of ``T`` objects, and that the caller wants
    to iterate over results as they become available.

    ``IterableModel`` is not instantiated directly — it carries typing metadata
    and provides schema/parsing helpers.

    Example::

        result = await shield.generate(prompt, output_type=IterableModel[Entity])
        # result.parsed is a list[Entity]
    """

    _wrapped_type: type[Any]

    def __class_getitem__(cls, item: type[T]) -> type[IterableModel[T]]:  # type: ignore[override]
        """Support ``IterableModel[T]`` subscript syntax."""
        new_cls: type[IterableModel[T]] = type(  # type: ignore[assignment]
            f"IterableModel[{getattr(item, '__name__', str(item))}]",
            (cls,),
            {"_wrapped_type": item},
        )
        return new_cls

    @classmethod
    def build_schema(cls, model_type: type[Any]) -> dict[str, Any]:
        """Build a JSON schema for an array of ``model_type`` objects.

        Args:
            model_type: The Pydantic model or type for each item in the array.

        Returns:
            JSON schema dict with ``type: "array"`` and the item schema.

        Example::

            schema = IterableModel.build_schema(Entity)
            # {"type": "array", "items": {"type": "object", "properties": {...}}}
        """
        if hasattr(model_type, "model_json_schema"):
            item_schema: dict[str, Any] = model_type.model_json_schema()
        else:
            item_schema = {"type": "object"}

        return {
            "type": "array",
            "items": item_schema,
            "description": f"Array of {getattr(model_type, '__name__', str(model_type))} objects.",
        }

    @classmethod
    def parse_items(cls, raw: str, model_type: type[Any]) -> list[Any]:
        """Parse a JSON array string into a list of ``model_type`` instances.

        Skips invalid items rather than failing the whole parse — robustness
        matters more than strictness for streaming use cases.

        Args:
            raw: Raw string from the backend, expected to be a JSON array.
            model_type: The Pydantic model class or callable for each item.

        Returns:
            List of successfully parsed ``model_type`` instances. Empty list
            if ``raw`` is not a valid JSON array.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        if not isinstance(data, list):
            return []

        results: list[Any] = []
        for item in data:
            try:
                if hasattr(model_type, "model_validate"):
                    results.append(model_type.model_validate(item))
                elif callable(model_type):
                    parsed = model_type(**item) if isinstance(item, dict) else model_type(item)
                    results.append(parsed)
                else:
                    results.append(item)
            except Exception:
                logger.debug("IterableModel: skipping invalid item %r", item)

        return results

    @classmethod
    def iter_items(cls, raw: str, model_type: type[Any]) -> Iterator[Any]:
        """Yield parsed items one by one from a JSON array string.

        Convenience wrapper around :meth:`parse_items` for use in for-loops.

        Args:
            raw: Raw string from the backend.
            model_type: The Pydantic model class for each item.

        Yields:
            Individual ``model_type`` instances.
        """
        yield from cls.parse_items(raw, model_type)

    @classmethod
    def get_wrapped_type(cls) -> type[Any]:
        """Return the wrapped type ``T`` from ``IterableModel[T]``.

        Returns:
            The type argument passed to ``IterableModel[T]``.

        Raises:
            AttributeError: If called on the base ``IterableModel`` class (not subscripted).
        """
        return cls._wrapped_type  # type: ignore[attr-defined]
