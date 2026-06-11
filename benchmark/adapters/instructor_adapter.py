"""Instructor baseline — Pydantic-validated structured output with a retry loop.

Instructor's mechanism is a single call whose response is validated against a
Pydantic model, re-asking on validation failure. To run it against an arbitrary
JSON Schema we build a permissive Pydantic model from the schema (every field
optional, so partial extraction is not rejected outright). The model is still a
single whole-schema call — no decomposition — which is the point of comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import _common

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ._base import AdapterOutput

_TEMPERATURE: float = 0.0
# How many retries Instructor may spend re-asking on a validation failure. Kept
# in step with the retry budget the other methods receive (fairness, §7).
_MAX_RETRIES: int = 1
# Nesting depth past which the converter stops building typed sub-models and
# falls back to a free-form object, bounding model-build cost on deep schemas.
_MAX_DEPTH: int = 6

_SCALAR_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


@dataclass(frozen=True, slots=True)
class InstructorAdapter:
    """Single Instructor call with a schema-derived Pydantic model.

    Args:
        api_key: Provider credential; ``None`` uses the SDK's env pickup.
        base_url: Optional provider base URL.
    """

    name: str = field(default="instructor", init=False)
    api_key: str | None = None
    base_url: str | None = None

    def run(
        self,
        document: str,
        schema: dict[str, Any],
        *,
        model: str,
        context_window: int,
        max_output_tokens: int,
        instructions: str = "",
    ) -> AdapterOutput:
        """Extract via Instructor's validated structured-output call."""
        import instructor

        started = time.perf_counter()
        try:
            response_model = _model_from_schema(schema, "Extraction", depth=0)
            client = instructor.from_groq(
                _common.groq_client(self.api_key, self.base_url), mode=instructor.Mode.JSON
            )
            result = client.chat.completions.create(
                model=_common.model_id(model),
                # SDK message params are TypedDicts; plain dicts are accepted at runtime.
                messages=_common.messages(  # type: ignore[arg-type]
                    document,
                    schema,
                    context_window=context_window,
                    max_output_tokens=max_output_tokens,
                    instructions=instructions,
                ),
                response_model=response_model,
                max_tokens=max_output_tokens,
                temperature=_TEMPERATURE,
                max_retries=_MAX_RETRIES,
            )
            data = result.model_dump(by_alias=True, exclude_none=False)
        except Exception as exc:  # record fairly, never abort the sweep
            return _common.failure_output(schema, round(time.perf_counter() - started, 3), exc)
        return _common.success_output(data, schema, round(time.perf_counter() - started, 3))


def _model_from_schema(node: dict[str, Any], name: str, *, depth: int) -> type[BaseModel]:
    from pydantic import ConfigDict, Field, create_model

    fields: dict[str, Any] = {}
    properties = node.get("properties", {})
    for index, (key, sub) in enumerate(properties.items() if isinstance(properties, dict) else []):
        annotation = _annotation(sub if isinstance(sub, dict) else {}, f"{name}_{index}", depth)
        safe = _safe_field(key, index)
        # When a key is not a valid identifier, keep the schema key as an alias so
        # the model still reads and writes it (gold paths align), and dump by alias.
        default = Field(default=None, alias=key) if safe != key else None
        fields[safe] = (annotation | None, default)
    return create_model(name, __config__=ConfigDict(populate_by_name=True), **fields)


def _annotation(node: dict[str, Any], name: str, depth: int) -> Any:
    if "enum" in node:
        return str
    node_type = node.get("type")
    if node_type == "object" and depth < _MAX_DEPTH:
        return _model_from_schema(node, name, depth=depth + 1)
    if node_type == "array":
        items = node.get("items")
        inner = _annotation(items, f"{name}_item", depth + 1) if isinstance(items, dict) else Any
        return list[inner]  # type: ignore[valid-type]
    if isinstance(node_type, str) and depth < _MAX_DEPTH:
        return _SCALAR_TYPES.get(node_type, Any)
    return Any


def _safe_field(key: str, index: int) -> str:
    # pydantic create_model needs valid identifiers; a non-identifier key (e.g.
    # "0_14_years") gets a positional fallback name, with the original key kept as
    # a Field alias by the caller so output and gold paths still align.
    return key if key.isidentifier() and not key.startswith("_") else f"field_{index}"
