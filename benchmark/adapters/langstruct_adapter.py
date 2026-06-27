"""LangStruct competitor - DSPy-based extraction over the same Groq model.

LangStruct (https://langstruct.dev) takes a Pydantic schema and runs its own
DSPy/LiteLLM chunk-and-merge extraction, so it accepts the shared model id
("groq/...") directly. It extracts reliably for flat schemas but collapses to an
empty result on a deeply-nested wide schema, so - to compare on the *same* target
fields, not on schema shape - we flatten the JSON Schema's leaves to a flat
Pydantic model (one typed field per leaf, keyed by its full path) and re-nest the
extracted values back into the schema shape for scoring.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import Field, create_model

from . import _common

if TYPE_CHECKING:
    from ._base import AdapterOutput

_USE_SOURCES: bool = False  # we score values only; source spans add cost, not score
_PRIMITIVES: dict[str, type] = {"string": str, "integer": int, "number": float, "boolean": bool}
_model_counter = itertools.count()
_PATH_SEP = "__"  # joins a leaf's path into one valid identifier (dots are illegal)
# LangStruct chunks a document and calls the model per chunk; on a wide schema over
# a large document that fans out into many calls. Without retries a shared TPM 429
# silently drops a chunk's result, mis-scoring the method as 0. Match the SDK
# baselines' transient-retry budget so a rate-limit storm penalizes no method.
_NUM_RETRIES = _common.MAX_TRANSIENT_RETRIES


def _leaf_paths(
    node: dict[str, Any], prefix: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], type]]:
    """Flatten a JSON Schema to ``(path, python_type)`` pairs, one per leaf.

    Mirrors ``schema_field_count``: nested objects recurse; arrays and scalars are
    single leaves. The path is the full key chain, so leaves with the same local
    name in different branches stay distinct.
    """
    out: list[tuple[tuple[str, ...], type]] = []
    for key, child in node.get("properties", {}).items():
        if isinstance(child, dict) and child.get("type") == "object":
            out.extend(_leaf_paths(child, (*prefix, key)))
        else:
            child_type = child.get("type", "") if isinstance(child, dict) else ""
            py_type = list if child_type == "array" else _PRIMITIVES.get(child_type, str)
            out.append(((*prefix, key), py_type))
    return out


def _flat_model(leaves: list[tuple[tuple[str, ...], type]]) -> Any:
    """Build a flat Pydantic model: one optional field per leaf, keyed by its path."""
    fields: dict[str, Any] = {}
    for path, py_type in leaves:
        name = _PATH_SEP.join(path)
        fields[name] = (py_type | None, Field(default=None, description=".".join(path)))
    model_name = f"Flat_{next(_model_counter)}"
    return create_model(model_name, **fields)


def _nest(leaves: list[tuple[tuple[str, ...], type]], entities: dict[str, Any]) -> dict[str, Any]:
    """Re-nest the flat extracted values back into the schema's object shape."""
    nested: dict[str, Any] = {}
    for path, _ in leaves:
        value = entities.get(_PATH_SEP.join(path))
        if value in (None, "", [], {}):
            continue
        cursor = nested
        for segment in path[:-1]:
            cursor = cursor.setdefault(segment, {})
        cursor[path[-1]] = value
    return nested


@dataclass(frozen=True, slots=True)
class LangStructAdapter:
    """LangStruct extraction (its own chunk-and-merge) on the shared Groq model."""

    name: str = field(default="langstruct", init=False)
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
        """Extract via LangStruct over a flat view of the schema, then re-nest."""
        import dspy
        from langstruct import LangStruct

        started = time.perf_counter()
        try:
            # Fit the document to the shared input window before LangStruct chunks it -
            # the same budget rule every competitor adapter applies (_common).
            fitted = _common._fit_document(document, context_window, max_output_tokens)
            leaves = _leaf_paths(schema)
            pydantic_schema = _flat_model(leaves)
            # Build the LM with the shared output budget and a transient-retry budget:
            # max_tokens must be set or wide-schema responses truncate and fail to parse
            # (DSPy defaults it to None); retries stop a shared 429 from zeroing a chunk.
            language_model = dspy.LM(
                model,
                temperature=0.0,
                max_tokens=max_output_tokens,
                num_retries=_NUM_RETRIES,
            )
            extractor = LangStruct(
                model=language_model, schema=pydantic_schema, use_sources=_USE_SOURCES
            )
            result = extractor.extract(fitted)
            entities = result.entities if isinstance(result.entities, dict) else {}
            data = _nest(leaves, entities)
        except Exception as exc:  # record fairly, never abort the sweep
            return _common.failure_output(schema, round(time.perf_counter() - started, 3), exc)
        return _common.success_output(data, schema, round(time.perf_counter() - started, 3))
