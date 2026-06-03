"""FormatShield assembly module — JSON assembly, blackboard, and quality scoring.

Public surface
--------------
* :func:`assemble_json` — assemble flat SFEP results into a nested JSON dict.
* :func:`parse_path_segments` — parse dot-notation path into segment list.
* :class:`RadixTrie` — low-level trie for custom assembly workflows.
* :class:`Blackboard` — per-field state machine for extraction state tracking.
* :class:`FieldState` — enum of the 6 blackboard field states.
* :func:`compute_quality_score` — compute quality metrics from blackboard state.
* :class:`QualityReport` — immutable quality metrics dataclass.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._blackboard import Blackboard, FieldState
    from ._quality import QualityReport, compute_quality_score
    from ._trie import RadixTrie, assemble_json, parse_path_segments

__all__ = [
    "Blackboard",
    "FieldState",
    "QualityReport",
    "RadixTrie",
    "assemble_json",
    "compute_quality_score",
    "parse_path_segments",
]

_dynamic_imports: dict[str, str] = {
    "Blackboard": "._blackboard",
    "FieldState": "._blackboard",
    "QualityReport": "._quality",
    "compute_quality_score": "._quality",
    "RadixTrie": "._trie",
    "assemble_json": "._trie",
    "parse_path_segments": "._trie",
}


def __getattr__(name: str) -> object:
    if name in _dynamic_imports:
        module = importlib.import_module(_dynamic_imports[name], package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
