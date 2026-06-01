"""Unit test fixtures — mock providers, fake schemas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def make_schema() -> Callable[..., dict]:  # type: ignore[type-arg]
    """Factory fixture for creating minimal JSON Schema dicts."""

    def _make(
        properties: dict | None = None,  # type: ignore[type-arg]
        required: list[str] | None = None,
    ) -> dict:  # type: ignore[type-arg]
        return {
            "type": "object",
            "properties": properties or {"name": {"type": "string"}},
            "required": required or [],
        }

    return _make  # type: ignore[return-value]


@pytest.fixture
def simple_string_schema() -> dict:  # type: ignore[type-arg]
    """Single required string field schema."""
    return {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }


@pytest.fixture
def multi_type_schema() -> dict:  # type: ignore[type-arg]
    """Schema covering all 8 primitive field types."""
    return {
        "type": "object",
        "properties": {
            "flag": {"type": "boolean"},
            "count": {"type": "integer"},
            "amount": {"type": "number"},
            "label": {"type": "string"},
            "status": {"type": "string", "enum": ["active", "inactive", "pending"]},
            "nothing": {"type": "null"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "nested": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
            },
        },
        "required": ["flag", "count"],
    }
