"""FormatShield schema analysis module.

Provides pure-function schema flattening, token prediction, difficulty scoring,
and dependency extraction. Zero API calls, zero side effects.
"""

from __future__ import annotations

from ._deps import extract_dependencies as extract_dependencies
from ._difficulty import compute_difficulty as compute_difficulty
from ._flatten import flatten_schema as flatten_schema
from ._preflight import preflight_schema as preflight_schema
from ._tau import compute_tau as compute_tau
from ._types import (
    SEGMENT_TYPE_STRUCTURED as SEGMENT_TYPE_STRUCTURED,
)
from ._types import (
    SEGMENT_TYPE_TABULAR as SEGMENT_TYPE_TABULAR,
)
from ._types import (
    SEGMENT_TYPE_UNSTRUCTURED as SEGMENT_TYPE_UNSTRUCTURED,
)
from ._types import (
    CapacityLeaf as CapacityLeaf,
)
from ._types import (
    Field as Field,
)
from ._types import (
    FieldGroup as FieldGroup,
)
from ._types import (
    Segment as Segment,
)

__all__ = [
    "SEGMENT_TYPE_STRUCTURED",
    "SEGMENT_TYPE_TABULAR",
    "SEGMENT_TYPE_UNSTRUCTURED",
    "CapacityLeaf",
    "Field",
    "FieldGroup",
    "Segment",
    "compute_difficulty",
    "compute_tau",
    "extract_dependencies",
    "flatten_schema",
    "preflight_schema",
]
