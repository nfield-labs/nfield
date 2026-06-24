"""Tabular export of extraction results to pandas / CSV.

pandas is an optional dependency: import only when an export function is called, and
raise a clear install hint if it is absent, so the core library stays dependency-free
(``[project.optional-dependencies] export``). Each result becomes one row keyed by the
flat dot-notation field paths (the natural columnar shape for N-field extraction),
with optional metadata columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import pandas as pd

    from formatshield.types import ExtractionResult

__all__ = [
    "result_to_dataframe",
    "results_to_csv",
    "results_to_dataframe",
]

_PANDAS_HINT: str = (
    "pandas is required for DataFrame/CSV export. Install it with: "
    "pip install 'formatshield[export]'"
)
# Metadata columns added when include_metadata=True. Prefixed so they never collide
# with a schema field path of the same name.
_META_PREFIX: str = "_meta."


def _require_pandas() -> Any:
    """Import pandas on demand, raising a clear install hint when it is absent.

    Returns:
        The imported ``pandas`` module.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(_PANDAS_HINT) from exc
    return pd


def _result_row(result: ExtractionResult, *, include_metadata: bool) -> dict[str, Any]:
    """Flatten one result to a single ``{column: value}`` row.

    Uses the flat per-field results (dot-notation path → value) when present, falling
    back to ``data`` otherwise. Metadata columns are added under :data:`_META_PREFIX`.

    Args:
        result: The result to flatten.
        include_metadata: Whether to append run-level metadata columns.

    Returns:
        A row dict suitable for a pandas DataFrame.
    """
    row: dict[str, Any] = (
        {f.path: f.value for f in result.fields} if result.fields else dict(result.data)
    )
    if include_metadata:
        row[f"{_META_PREFIX}status"] = result.status.value
        row[f"{_META_PREFIX}quality_score"] = result.metadata.quality_score
        row[f"{_META_PREFIX}fields_extracted"] = result.metadata.fields_extracted
        row[f"{_META_PREFIX}fields_total"] = result.metadata.fields_total
    return row


def results_to_dataframe(
    results: Iterable[ExtractionResult],
    *,
    include_metadata: bool = False,
) -> pd.DataFrame:
    """Convert results to a pandas DataFrame, one row per result.

    Args:
        results: The results to tabulate.
        include_metadata: Add run-level metadata columns (status, quality score,
            field counts) under a ``_meta.`` prefix. Default ``False``.

    Returns:
        A DataFrame whose columns are the union of the results' field paths.

    Raises:
        ImportError: If pandas is not installed.

    Example:
        >>> # df = results_to_dataframe(batch, include_metadata=True)
    """
    pd = _require_pandas()
    rows = [_result_row(r, include_metadata=include_metadata) for r in results]
    return pd.DataFrame(rows)


def result_to_dataframe(
    result: ExtractionResult,
    *,
    include_metadata: bool = False,
) -> pd.DataFrame:
    """Convert a single result to a one-row pandas DataFrame.

    Args:
        result: The result to tabulate.
        include_metadata: Add run-level metadata columns. Default ``False``.

    Returns:
        A one-row DataFrame.

    Raises:
        ImportError: If pandas is not installed.

    Example:
        >>> # df = result_to_dataframe(result)
    """
    return results_to_dataframe([result], include_metadata=include_metadata)


def results_to_csv(
    results: Iterable[ExtractionResult],
    path: str | Path,
    *,
    include_metadata: bool = False,
) -> None:
    """Write results to a CSV file, one row per result.

    Args:
        results: The results to write.
        path: Destination ``.csv`` path.
        include_metadata: Add run-level metadata columns. Default ``False``.

    Raises:
        ImportError: If pandas is not installed.

    Example:
        >>> # results_to_csv(batch, "out/results.csv")
    """
    frame = results_to_dataframe(results, include_metadata=include_metadata)
    frame.to_csv(path, index=False)
