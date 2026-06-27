"""ContextGem competitor - concept-based extraction over the same Groq model.

ContextGem (https://github.com/shcherbak-ai/contextgem) models extraction as a set
of *concepts* attached to a Document: each concept carries a name + description and
is filled by its own LLM reasoning. It drives the model through litellm, so it takes
the shared model id ("groq/...") directly. We map each JSON Schema leaf to a
``StringConcept`` (keyed by its full path), run extraction, and re-nest the concept
values into the schema's shape so the scorer counts coverage fairly.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import _common

if TYPE_CHECKING:
    from ._base import AdapterOutput

_PATH_SEP = "."  # full dotted path is the unique concept name and the re-nest key
# A wide schema becomes many concepts, each its own reasoning pass; give the LM a
# transient-retry budget so a shared 429 storm penalizes no method (fairness).
_NUM_RETRIES = _common.MAX_TRANSIENT_RETRIES


@dataclass(frozen=True, slots=True)
class ContextGemAdapter:
    """ContextGem concept extraction, mapped onto the schema, on the shared Groq model."""

    name: str = field(default="contextgem", init=False)
    api_key: str | None = None

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
        """Extract one concept per schema leaf, then re-nest into the schema shape."""
        from contextgem import Document, DocumentLLM, StringConcept

        started = time.perf_counter()
        try:
            fitted = _common._fit_document(document, context_window, max_output_tokens)
            leaves = _leaf_paths(schema)
            document_llm = DocumentLLM(
                model=model,
                api_key=self.api_key or os.environ.get("GROQ_API_KEY"),
                max_tokens=max_output_tokens,
                num_retries_failed_request=_NUM_RETRIES,
            )
            doc = Document(raw_text=fitted)
            doc.concepts = [
                StringConcept(
                    name=_PATH_SEP.join(path),
                    description=description or _PATH_SEP.join(path),
                    singular_occurrence=True,
                )
                for path, description in leaves
            ]
            document_llm.extract_concepts_from_document(doc)
            got = {
                concept.name: concept.extracted_items[0].value
                for concept in doc.concepts
                if concept.extracted_items
            }
            data = _nest([p for p, _ in leaves], got)
        except Exception as exc:  # record fairly, never abort the sweep
            return _common.failure_output(schema, round(time.perf_counter() - started, 3), exc)
        return _common.success_output(data, schema, round(time.perf_counter() - started, 3))


def _leaf_paths(
    node: dict[str, Any], prefix: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], str]]:
    """Flatten the schema to ``(path, description)`` pairs, one per leaf."""
    out: list[tuple[tuple[str, ...], str]] = []
    for key, child in node.get("properties", {}).items():
        if isinstance(child, dict) and child.get("type") == "object":
            out.extend(_leaf_paths(child, (*prefix, key)))
        else:
            description = child.get("description", "") if isinstance(child, dict) else ""
            out.append(((*prefix, key), description))
    return out


def _nest(paths: list[tuple[str, ...]], got: dict[str, Any]) -> dict[str, Any]:
    """Re-nest the flat concept values back into the schema's object shape."""
    nested: dict[str, Any] = {}
    for path in paths:
        value = got.get(_PATH_SEP.join(path))
        if value in (None, "", [], {}):
            continue
        cursor = nested
        for segment in path[:-1]:
            cursor = cursor.setdefault(segment, {})
        cursor[path[-1]] = value
    return nested
