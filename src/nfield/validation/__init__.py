"""nfield validation module - field validation and surgical retry.

Public surface
--------------
* :func:`validate_field` - validate a value against its field's type and constraints.
* :func:`constraint_check` - return all constraint violations for a value.
* :func:`classify_failure` - classify the root cause of a field failure.
* :func:`orchestrate_retry` - run up to 2 rounds of surgical field retry.
* :func:`surgical_field_retry` - execute one targeted retry API call.
* :func:`build_retry_prompt` - build the messages list for a retry call.
* :func:`split_retry_batches` - group failed fields by dependency closure.
* :func:`handle_missing_fields` - tree backtrack for absent fields.
* :class:`FailureCause` - enum of the 4 failure causes.
* :func:`grounding_score` / :func:`is_grounded` / :func:`is_groundable` - score whether
  an extracted value is supported by the source text (anti-hallucination).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._grounding import grounding_score, is_groundable, is_grounded
    from ._retry import (
        FailureCause,
        build_retry_prompt,
        classify_failure,
        handle_missing_fields,
        orchestrate_retry,
        split_retry_batches,
        surgical_field_retry,
    )
    from ._type_check import constraint_check, validate_field

__all__ = [
    "FailureCause",
    "build_retry_prompt",
    "classify_failure",
    "constraint_check",
    "grounding_score",
    "handle_missing_fields",
    "is_groundable",
    "is_grounded",
    "orchestrate_retry",
    "split_retry_batches",
    "surgical_field_retry",
    "validate_field",
]

_dynamic_imports: dict[str, str] = {
    "FailureCause": "._retry",
    "build_retry_prompt": "._retry",
    "classify_failure": "._retry",
    "handle_missing_fields": "._retry",
    "orchestrate_retry": "._retry",
    "split_retry_batches": "._retry",
    "surgical_field_retry": "._retry",
    "constraint_check": "._type_check",
    "validate_field": "._type_check",
    "grounding_score": "._grounding",
    "is_grounded": "._grounding",
    "is_groundable": "._grounding",
}


def __getattr__(name: str) -> object:
    if name in _dynamic_imports:
        module = importlib.import_module(_dynamic_imports[name], package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
