"""NField command-line interface.

Two commands:

* ``nfield extract`` — run the extraction pipeline on a document and print
  the resulting JSON.
* ``nfield inspect`` — analyse a schema offline (no API calls): field
  count, type breakdown, and a minimum-call (K_min) estimate.

The CLI is an optional extra: install with ``pip install "nfield[cli]"``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from nfield import __version__, nfield
from nfield.config import ExtractionConfig
from nfield.exceptions import NFieldError
from nfield.pipeline.s2c_packing import compute_K_min
from nfield.schema._flatten import flatten_schema
from nfield.schema._tau import compute_tau

__all__ = ["app", "main"]

# Offline estimation constants for `inspect`. These mirror the pipeline's
# defaults so the estimate tracks a real run without contacting a provider:
#   - NSL (chars/token) fallback for English (arXiv:2411.12240)
#   - a conservative 8K output ceiling
#   - the heavy-tail-inflated z-score used by capacity packing (CASTILLO,
#     arXiv:2505.16881): z_target(95th pct) x 1.5
_INSPECT_CHARS_PER_TOKEN: float = 3.5
_INSPECT_OUTPUT_CEILING: int = 8_192
_INSPECT_Z_EFF: float = 1.645 * 1.5
_PATH_PREVIEW_LIMIT: int = 20

app = typer.Typer(
    name="nfield",
    help="N-field structured extraction from documents with LLMs.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    """Print the installed version and exit when ``--version`` is passed."""
    if value:
        typer.echo(f"nfield {__version__}")
        raise typer.Exit


@app.callback()
def _root(
    _version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """NField — extract N structured fields from any document."""


def _read_text_file(path: Path, label: str) -> str:
    """Read a UTF-8 text file, turning every I/O failure into a clean message.

    Centralises the ways a user-supplied path can fail — missing, a directory,
    permission-denied, or not UTF-8 — so the CLI reports a one-line
    ``BadParameter`` instead of a Python traceback.

    Args:
        path: File to read.
        label: Human label for the file (e.g. ``"Document"``, ``"Schema"``),
            used in the error message.

    Returns:
        The file's decoded text.

    Raises:
        typer.BadParameter: If the path is missing, not UTF-8, or otherwise
            unreadable (directory, permissions, ...).

    Example:
        >>> # _read_text_file(Path("missing.txt"), "Document")  # → BadParameter
        True
    """
    if not path.exists():
        raise typer.BadParameter(f"{label} file not found: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise typer.BadParameter(f"{label} is not valid UTF-8 text: {path}") from exc
    except OSError as exc:
        raise typer.BadParameter(f"Could not read {label.lower()} file {path}: {exc}") from exc


def _load_schema(schema_path: Path) -> dict[str, Any]:
    """Load and parse a JSON Schema file.

    Args:
        schema_path: Path to a ``.json`` file containing a JSON Schema.

    Returns:
        The parsed schema dict.

    Raises:
        typer.BadParameter: If the file is missing, unreadable, not valid JSON,
            or not a JSON object.
    """
    try:
        loaded: Any = json.loads(_read_text_file(schema_path, "Schema"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Schema is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise typer.BadParameter("Schema must be a JSON object.")
    return loaded


@app.command()
def extract(
    document: Annotated[Path, typer.Argument(help="Path to the source document (text).")],
    schema: Annotated[Path, typer.Option("--schema", "-s", help="Path to a JSON Schema file.")],
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model string, e.g. groq/llama-3.1-8b-instant.")
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write JSON here instead of stdout."),
    ] = None,
    max_retry_rounds: Annotated[
        int, typer.Option("--max-retry-rounds", help="Validation retry rounds.")
    ] = 2,
    context_window: Annotated[
        int | None,
        typer.Option("--context-window", help="Model's real context window (tokens)."),
    ] = None,
    max_output_tokens: Annotated[
        int | None,
        typer.Option("--max-output-tokens", help="Model's real max output tokens."),
    ] = None,
    instructions: Annotated[
        str, typer.Option("--instructions", help="Extra steering for the model, prepended.")
    ] = "",
) -> None:
    """Extract structured fields from a document into the given schema.

    Reads the document and schema, runs the pipeline against *model*, and writes
    the extracted JSON to stdout (or ``--output``).
    """
    schema_dict = _load_schema(schema)
    document_text = _read_text_file(document, "Document")

    config = ExtractionConfig(max_retry_rounds=max_retry_rounds)
    try:
        result = nfield(
            document_text,
            schema_dict,
            model,
            config=config,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            instructions=instructions,
        )
    except NFieldError as exc:
        typer.echo(f"Extraction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    payload = json.dumps(result.data, indent=2, ensure_ascii=False)
    if output is not None:
        try:
            output.write_text(payload + "\n", encoding="utf-8")
        except OSError as exc:
            typer.echo(f"Could not write output to {output}: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"Wrote {result.metadata.fields_extracted} fields to {output}", err=True)
    else:
        typer.echo(payload)


@app.command()
def inspect(
    schema: Annotated[Path, typer.Argument(help="Path to a JSON Schema file.")],
    max_output_tokens: Annotated[
        int,
        typer.Option("--max-output-tokens", help="Model's output ceiling (M_O) for K_min."),
    ] = _INSPECT_OUTPUT_CEILING,
) -> None:
    """Analyse a schema offline: field count, type breakdown, K_min estimate.

    Makes no API calls. K_min depends on the model's output ceiling (M_O); pass
    ``--max-output-tokens`` for your model to get a faithful estimate (it
    defaults to a conservative 8K). The English NSL ratio is assumed for tau.
    """
    schema_dict = _load_schema(schema)
    fields = flatten_schema(schema_dict)
    if not fields:
        typer.echo("Schema produced zero extractable fields.", err=True)
        raise typer.Exit(code=1)

    enriched = []
    for f in fields:
        tau, var_tau = compute_tau(f, _INSPECT_CHARS_PER_TOKEN)
        enriched.append(f.with_tau(tau=tau, var_tau=var_tau))
    sum_var = sum(f.var_tau for f in enriched)
    # Mirrors the safe-output reservation in pipeline/s2c_packing (M_O minus a
    # heavy-tail margin); kept in sync by hand since this is an offline estimate,
    # not the real packing run. compute_K_min treats safe_output <= 0 as len(fields).
    safe_output = max_output_tokens - _INSPECT_Z_EFF * math.sqrt(sum_var)
    k_min = compute_K_min(enriched, safe_output, _INSPECT_CHARS_PER_TOKEN)

    type_counts: dict[str, int] = {}
    for f in fields:
        type_counts[f.type] = type_counts.get(f.type, 0) + 1

    typer.echo(f"Schema: {schema}")
    typer.echo(f"Total leaf fields : {len(fields)}")
    typer.echo(f"K_min estimate    : {k_min}  (M_O={max_output_tokens}, English NSL)")
    typer.echo("Field types:")
    for type_name, count in sorted(type_counts.items()):
        typer.echo(f"  {type_name:<10} {count}")

    typer.echo("Paths:")
    for f in fields[:_PATH_PREVIEW_LIMIT]:
        typer.echo(f"  {f.path}")
    if len(fields) > _PATH_PREVIEW_LIMIT:
        typer.echo(f"  ... and {len(fields) - _PATH_PREVIEW_LIMIT} more")


def main() -> None:
    """Entry point for the ``nfield`` console script."""
    app()


if __name__ == "__main__":
    sys.exit(app())  # pragma: no cover
