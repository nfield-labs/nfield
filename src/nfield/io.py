"""Filesystem helpers: load a document or schema from disk, save/load results.

The library API takes a document ``str`` and a schema ``dict``; these helpers do
the file reading so callers do not hand-roll it (the same step the CLI does). Input
loading is text/JSON only — no PDF/DOCX/CSV parsing, which stays the caller's job —
keeping the "text in, structured out" boundary. Results are persisted as JSON Lines
(one result per line), the de-facto format for a stream of records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nfield.exceptions import SchemaError
from nfield.types import ExtractionResult

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "load_document",
    "load_results",
    "load_schema",
    "save_results",
]


def load_document(path: str | Path) -> str:
    """Read a UTF-8 text document from disk.

    Args:
        path: Path to a ``.txt`` / ``.md`` / any UTF-8 text file.

    Returns:
        The file's decoded text, ready to pass as the ``document`` argument.

    Raises:
        FileNotFoundError: If the path does not exist.
        UnicodeDecodeError: If the file is not valid UTF-8 (it is not text).

    Example:
        >>> # document = load_document("contract.md")
    """
    return Path(path).read_text(encoding="utf-8")


def load_schema(path: str | Path) -> dict[str, Any]:
    """Read and parse a JSON Schema file into the dict the engine consumes.

    Args:
        path: Path to a ``.json`` file holding a JSON Schema object.

    Returns:
        The parsed schema dict, ready to pass as the ``schema`` argument.

    Raises:
        FileNotFoundError: If the path does not exist.
        SchemaError: If the file is not valid JSON, or its top level is not an object.

    Example:
        >>> # schema = load_schema("invoice_schema.json")
    """
    try:
        loaded: Any = json.loads(load_document(path))
    except json.JSONDecodeError as exc:
        raise SchemaError(
            f"Schema file {Path(path).name!r} is not valid JSON: {exc}",
            hint="Check the JSON syntax (trailing commas, unquoted keys).",
        ) from exc
    if not isinstance(loaded, dict):
        raise SchemaError(
            "Schema file must contain a JSON object at the top level.",
            hint="The root must be an object ({...}), not a list or scalar.",
        )
    return loaded


def save_results(results: Iterable[ExtractionResult], path: str | Path) -> None:
    """Write extraction results to a JSON Lines file, one result per line.

    Parent directories are created if absent. Each line is a complete JSON object
    from :meth:`ExtractionResult.to_dict`, so the file streams record-by-record and
    round-trips with :func:`load_results`.

    Args:
        results: The results to persist (a single batch or an accumulated stream).
        path: Destination ``.jsonl`` path.

    Example:
        >>> # save_results(batch, "out/results.jsonl")
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def load_results(path: str | Path) -> list[ExtractionResult]:
    """Read results back from a JSON Lines file written by :func:`save_results`.

    Args:
        path: Source ``.jsonl`` path.

    Returns:
        The reconstructed results, in file order. Blank lines are skipped.

    Raises:
        FileNotFoundError: If the path does not exist.

    Example:
        >>> # results = load_results("out/results.jsonl")
    """
    results: list[ExtractionResult] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                results.append(ExtractionResult.from_dict(json.loads(stripped)))
    return results
