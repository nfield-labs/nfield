"""Stage 2A: Structural Grouping.

Zero API calls. Groups fields by their parent dot-notation path, creating
one FieldGroup per unique parent. Top-level fields share a group under
the empty string parent. Output: state.groups, state.group_map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from formatshield.schema._types import FieldGroup

if TYPE_CHECKING:
    from formatshield.pipeline._state import PipelineState

__all__ = ["run_stage_2a"]

# Top-level fields (no parent) get this group key
_TOP_LEVEL_GROUP_KEY: str = ""


def run_stage_2a(state: PipelineState) -> PipelineState:
    """Group fields by shared parent path.

    MVP algorithm: parent-path grouping — all fields sharing the same
    ``field.parent_path`` are placed in one ``FieldGroup``.

    Top-level fields (parent_path == "") form their own group.

    Populates:
    - ``state.groups`` — list of ``FieldGroup`` objects
    - ``state.group_map`` — field path → its ``FieldGroup``

    Args:
        state: Pipeline state from Stage 1 (must have ``state.fields`` set).

    Returns:
        Updated ``PipelineState``.

    Example:
        >>> callable(run_stage_2a)
        True
    """
    group_by_parent: dict[str, FieldGroup] = {}

    for f in state.fields:
        parent = f.parent_path
        if parent not in group_by_parent:
            group_by_parent[parent] = FieldGroup(parent_path=parent)
        group_by_parent[parent].fields.append(f)

    groups = list(group_by_parent.values())
    group_map = {f.path: group_by_parent[f.parent_path] for f in state.fields}

    state.groups = groups
    state.group_map = group_map
    return state
