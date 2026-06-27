"""LangExtract competitor - Google's span-grounded entity extractor on Groq.

LangExtract (https://github.com/google/langextract) is entity/span oriented: given
a prompt and few-shot examples it returns a flat list of ``Extraction`` objects
(an ``extraction_class`` + the literal source span). It has no Groq provider, so we
drive it through its OpenAI-compatible provider pointed at the Groq endpoint, on the
same model and budget as every other method. Each schema leaf's full dotted path is
an extraction class (so leaves that share a local name in different branches stay
distinct); the returned classes are re-nested onto the schema's leaf shape so the
scorer counts coverage fairly.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import _common

if TYPE_CHECKING:
    import langextract as lx

    from ._base import AdapterOutput

_GROQ_OPENAI_BASE = "https://api.groq.com/openai/v1"
# Bigger buffer than the 1000-char default: fewer chunks => fewer per-chunk calls,
# so a wide schema over a large document does not fan out into a 429 storm.
_MAX_CHAR_BUFFER = 6000
# Modest parallelism keeps us under the shared TPM window (fairness, design §7).
_MAX_WORKERS = 4
_MAX_EXAMPLE_LEAVES = 8
# Listing every class name in the prompt is what steers an entity extractor at a
# fixed schema; cap it so the instruction stays well within the input window.
_MAX_PROMPT_CLASSES = 80
_PATH_SEP = "."  # full dotted path is the unique extraction class and the re-nest key


@dataclass(frozen=True, slots=True)
class LangExtractAdapter:
    """LangExtract entity extraction, mapped onto the schema, on the shared Groq model."""

    name: str = field(default="langextract", init=False)
    api_key: str | None = None
    base_url: str = _GROQ_OPENAI_BASE

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
        """Extract entities for the schema's leaf names and fold them into its shape."""
        import langextract as lx
        from langextract.providers.openai import OpenAILanguageModel

        started = time.perf_counter()
        try:
            fitted = _common._fit_document(document, context_window, max_output_tokens)
            paths = _leaf_paths(schema)
            classes = [_PATH_SEP.join(p) for p in paths]
            language_model = OpenAILanguageModel(
                model_id=_common.model_id(model),
                api_key=self.api_key or os.environ.get("GROQ_API_KEY"),
                base_url=self.base_url,
                temperature=0.0,
                max_workers=_MAX_WORKERS,
                max_tokens=max_output_tokens,  # shared output budget (fairness)
            )
            result = lx.extract(
                text_or_documents=fitted,
                prompt_description=_prompt(instructions, classes),
                examples=[_example(classes)],
                model=language_model,
                use_schema_constraints=False,  # OpenAI-compat endpoint: no Gemini schema
                fence_output=True,
                max_char_buffer=_MAX_CHAR_BUFFER,
                show_progress=False,
            )
            got = {ext.extraction_class: ext.extraction_text for ext in result.extractions}
            data = _nest(paths, got)
        except Exception as exc:  # record fairly, never abort the sweep
            return _common.failure_output(schema, round(time.perf_counter() - started, 3), exc)
        return _common.success_output(data, schema, round(time.perf_counter() - started, 3))


def _leaf_paths(node: dict[str, Any], prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Full dotted paths of the schema's leaves, mirroring ``schema_field_count``.

    Nested objects recurse; arrays and scalars are single leaves. The full path
    keeps leaves with the same local name in different branches distinct.
    """
    out: list[tuple[str, ...]] = []
    for key, child in node.get("properties", {}).items():
        if isinstance(child, dict) and child.get("type") == "object":
            out.extend(_leaf_paths(child, (*prefix, key)))
        else:
            out.append((*prefix, key))
    return out


def _prompt(instructions: str, classes: list[str]) -> str:
    """Prompt steering the extractor at our field paths as the extraction classes."""
    listed = ", ".join(classes[:_MAX_PROMPT_CLASSES])
    head = instructions or "Extract the requested fields from the document."
    return f"{head} Use these exact extraction classes: {listed}."


def _example(classes: list[str]) -> lx.data.ExampleData:
    """A synthetic example teaching the class=field-path pattern with aligned spans."""
    import langextract as lx

    sample = classes[:_MAX_EXAMPLE_LEAVES] or ["field"]
    pairs = [(name, f"value{i}") for i, name in enumerate(sample)]
    text = " ".join(f"{name} is {value}." for name, value in pairs)
    return lx.data.ExampleData(
        text=text,
        extractions=[
            lx.data.Extraction(extraction_class=name, extraction_text=value)
            for name, value in pairs
        ],
    )


def _nest(paths: list[tuple[str, ...]], got: dict[str, str]) -> dict[str, Any]:
    """Re-nest the flat class->value map (keyed by dotted path) onto the schema shape."""
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
