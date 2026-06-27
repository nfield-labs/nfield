"""nfield extraction module - SFEP parsing and prompt construction.

Public surface
--------------
* :func:`parse_sfep` - parse SFEP key=value LLM output into a typed dict.
* :func:`typecast` - cast a raw string to a Python type from a Field descriptor.
* :func:`parse_sfep_line` - parse a single SFEP line (streaming use).
* :func:`build_extraction_prompt` - build the messages list for provider.complete().
* :func:`build_retry_system_message` - build the retry messages list.
* :func:`select_template` - choose the prompt verbosity tier for a budget.
* :func:`describe_field` - render a single field description line.
* :data:`NEEDS_REVALIDATION` - sentinel returned when the LLM signals uncertainty.
* :class:`TemplateType` - enum for CONCISE / STANDARD / VERBOSE tiers.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._papt import TemplateType, describe_field, select_template
    from ._prompt import build_extraction_prompt, build_retry_system_message
    from ._sfep import NEEDS_REVALIDATION, parse_sfep, parse_sfep_line, typecast

__all__ = [
    "NEEDS_REVALIDATION",
    "TemplateType",
    "build_extraction_prompt",
    "build_retry_system_message",
    "describe_field",
    "parse_sfep",
    "parse_sfep_line",
    "select_template",
    "typecast",
]

_dynamic_imports: dict[str, str] = {
    "TemplateType": "._papt",
    "describe_field": "._papt",
    "select_template": "._papt",
    "build_extraction_prompt": "._prompt",
    "build_retry_system_message": "._prompt",
    "NEEDS_REVALIDATION": "._sfep",
    "parse_sfep": "._sfep",
    "parse_sfep_line": "._sfep",
    "typecast": "._sfep",
}


def __getattr__(name: str) -> object:
    if name in _dynamic_imports:
        module = importlib.import_module(_dynamic_imports[name], package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
