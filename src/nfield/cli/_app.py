"""nfield command-line interface.

Three commands:

* ``nfield extract`` - run the extraction pipeline on one document and write the
  result as JSON, JSON Lines, or CSV. Exposes the full ``ExtractionConfig``
  surface (grounding, provenance, reasoning-model handling, recovery, closed-book,
  and more) as flags, grouped into help panels.
* ``nfield batch`` - extract every document in a directory (or an explicit list of
  files) with one reused, calibrated engine, streaming the results to JSON Lines.
* ``nfield inspect`` - analyse a schema offline (no API calls): field count, type
  breakdown, and a minimum-call (K_min) estimate.

The CLI is an optional extra: install with ``pip install "nfield[cli]"``. CSV output
additionally needs the export extra (``pip install "nfield[cli,export]"``).
"""

from __future__ import annotations

import json
import math
import sys
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from nfield import __version__
from nfield.config import ExtractionConfig
from nfield.engine import NField
from nfield.exceptions import NFieldError
from nfield.io import save_results
from nfield.pipeline.s2c_packing import compute_K_min
from nfield.schema._flatten import flatten_schema
from nfield.schema._tau import compute_tau

if TYPE_CHECKING:
    from nfield.types import ExtractionResult

__all__ = ["app", "main"]

# Offline estimation constants for `inspect`. These mirror the pipeline's runtime
# defaults so the estimate tracks a real run without contacting a provider:
#   - the English characters-per-token fallback used across the pipeline
#     (pipeline/s2c_packing._FALLBACK_CHARS_PER_TOKEN)
#   - a conservative output ceiling when the model's real M_O is unknown
#   - the heavy-tail-inflated z-score capacity packing reserves output with
#     (CASTILLO, arXiv:2505.16881): the 95th-percentile z-score x 1.5
_INSPECT_CHARS_PER_TOKEN: float = 4.0
_INSPECT_OUTPUT_CEILING: int = 8_192
_INSPECT_Z_EFF: float = 1.645 * 1.5
_PATH_PREVIEW_LIMIT: int = 20

# Files a directory input contributes to `batch` when no explicit pattern is given.
_DEFAULT_BATCH_PATTERN: str = "*.txt"

# Help-panel labels group the many extract flags in `--help` so the common ones
# are not lost among the tuning knobs.
_PANEL_CONNECTION: str = "Model and connection"
_PANEL_OUTPUT: str = "Output"
_PANEL_TUNING: str = "Extraction tuning"
_PANEL_GROUNDING: str = "Grounding and provenance"
_PANEL_RECOVERY: str = "Reliability and recovery"
_PANEL_KNOWLEDGE: str = "Closed-book and knowledge"


class OutputFormat(str, Enum):
    """Serialization format for extraction results.

    Attributes:
        JSON: Pretty-printed JSON. For ``extract`` this is the extracted data
            object; for ``batch`` it is an array of data objects, in input order.
        JSONL: One compact JSON object per result (the full result: data,
            metadata, status, fields). The natural streaming format for ``batch``.
        CSV: One row per result, columns keyed by flat dot-notation field paths.
            Requires the ``export`` extra (pandas).
    """

    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"


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
    """nfield - extract N structured fields from any document."""


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def _read_text_file(path: Path, label: str) -> str:
    """Read a UTF-8 text file, turning every I/O failure into a clean message.

    Centralises the ways a user-supplied path can fail - missing, a directory,
    permission-denied, or not UTF-8 - so the CLI reports a one-line
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


def _parse_confidence(pairs: list[str] | None) -> dict[str, float] | None:
    """Parse repeated ``--confidence TIER=SCORE`` options into a threshold map.

    Args:
        pairs: Raw ``"TIER=SCORE"`` strings, or ``None`` when the flag was unused.

    Returns:
        A ``{tier: score}`` mapping, or ``None`` when no pairs were given (so the
        config default is left untouched).

    Raises:
        typer.BadParameter: If a pair is malformed or its score is not a float.
    """
    if not pairs:
        return None
    thresholds: dict[str, float] = {}
    for pair in pairs:
        tier, sep, raw = pair.partition("=")
        if not sep or not tier.strip():
            raise typer.BadParameter(f"--confidence expects TIER=SCORE, got: {pair!r}")
        try:
            thresholds[tier.strip()] = float(raw)
        except ValueError as exc:
            raise typer.BadParameter(f"--confidence score is not a number: {pair!r}") from exc
    return thresholds


def _build_config(
    *,
    context_utilization_ratio: float | None = None,
    max_retry_rounds: int | None = None,
    z_target: float | None = None,
    confidence: list[str] | None = None,
    document_language: str | None = None,
    reasoning_model: bool | None = None,
    chars_per_token: float | None = None,
    think_budget_min: int | None = None,
    think_budget_max: int | None = None,
    evidence_score_threshold: float | None = None,
    use_advanced_sfr: bool | None = None,
    inject_dependencies: bool | None = None,
    cascade_dependency_invalidation: bool | None = None,
    knowledge_fallback: bool | None = None,
    max_fields_per_call: int | None = None,
    max_concurrent_calls: int | None = None,
    max_api_retries: int | None = None,
    strict_validation: bool | None = None,
    recover_conflicts: bool | None = None,
    recover_call_failed: bool | None = None,
    validate_schema: bool | None = None,
    ground_values: bool | None = None,
    grounding_min_score: float | None = None,
    provenance: bool | None = None,
    fallback_model: str | None = None,
    closed_book: bool | None = None,
    self_consistency: bool | None = None,
) -> ExtractionConfig:
    """Build an ``ExtractionConfig`` from CLI flags, honouring config defaults.

    Every argument defaults to ``None`` meaning "not set on the command line"; only
    supplied values are passed through, so an unset flag inherits the library's own
    default instead of the CLI hard-coding (and drifting from) it.

    Args:
        context_utilization_ratio: Fraction of the context window used for chunks.
        max_retry_rounds: Validation retry rounds.
        z_target: Output-reservation z-score.
        confidence: Repeated ``TIER=SCORE`` confidence thresholds.
        document_language: BCP-47 tag of the document language.
        reasoning_model: Treat the model as a reasoning model (disable thinking).
        chars_per_token: Pin the characters-per-token ratio.
        think_budget_min: Lower bound of the thinking-phase token budget.
        think_budget_max: Upper bound of the thinking-phase token budget.
        evidence_score_threshold: Minimum evidence score for a chunk.
        use_advanced_sfr: Enable advanced Semantic Field Routing.
        inject_dependencies: Feed resolved dependency values into dependent leaves.
        cascade_dependency_invalidation: Flag dependents when an upstream value changes.
        knowledge_fallback: Fill absent fields from the model's own knowledge.
        max_fields_per_call: Per-leaf reliability budget in difficulty-weighted units.
        max_concurrent_calls: Max leaf calls in flight at once.
        max_api_retries: Per-call transient-failure retry budget.
        strict_validation: Validate values exactly as extracted (no lenient coercion).
        recover_conflicts: Re-extract conflicting fields in the recovery pass.
        recover_call_failed: Retry transiently-failed fields in the recovery pass.
        validate_schema: Reject a provably-unsatisfiable schema before any API call.
        ground_values: Label each value's support against the source.
        grounding_min_score: Minimum grounding score to count as supported.
        provenance: Attach source char offsets per value.
        fallback_model: Stronger model to escalate still-failing fields to.
        closed_book: Fill the schema from model knowledge with no document.
        self_consistency: Sample each closed-book leaf twice and keep agreeing values.

    Returns:
        A fully-formed :class:`~nfield.config.ExtractionConfig`.

    Raises:
        typer.BadParameter: If only one side of the thinking-phase budget is given,
            or a confidence pair is malformed.
    """
    if (think_budget_min is None) != (think_budget_max is None):
        raise typer.BadParameter("--think-budget-min and --think-budget-max must be set together.")

    # Assemble only the flags the user actually set; the rest fall through to the
    # ExtractionConfig defaults, so the CLI never re-declares (and drifts from) them.
    kwargs: dict[str, Any] = {}
    scalars: dict[str, Any] = {
        "context_utilization_ratio": context_utilization_ratio,
        "max_retry_rounds": max_retry_rounds,
        "z_target": z_target,
        "document_language": document_language,
        "reasoning_model": reasoning_model,
        "chars_per_token": chars_per_token,
        "evidence_score_threshold": evidence_score_threshold,
        "use_advanced_sfr": use_advanced_sfr,
        "inject_dependencies": inject_dependencies,
        "cascade_dependency_invalidation": cascade_dependency_invalidation,
        "knowledge_fallback": knowledge_fallback,
        "max_fields_per_call": max_fields_per_call,
        "max_concurrent_calls": max_concurrent_calls,
        "max_api_retries": max_api_retries,
        "strict_validation": strict_validation,
        "recover_conflicts": recover_conflicts,
        "recover_call_failed": recover_call_failed,
        "validate_schema": validate_schema,
        "ground_values": ground_values,
        "grounding_min_score": grounding_min_score,
        "provenance": provenance,
        "fallback_model": fallback_model,
        "closed_book": closed_book,
        "self_consistency": self_consistency,
    }
    kwargs.update({name: value for name, value in scalars.items() if value is not None})

    thresholds = _parse_confidence(confidence)
    if thresholds is not None:
        kwargs["confidence_thresholds"] = thresholds
    if think_budget_min is not None and think_budget_max is not None:
        kwargs["think_phase_budget"] = (think_budget_min, think_budget_max)

    return ExtractionConfig(**kwargs)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _result_csv_text(results: list[ExtractionResult]) -> str:
    """Render results as CSV text, raising a clean hint when pandas is absent.

    Args:
        results: The results to tabulate (one row each).

    Returns:
        The CSV document as a string, including the header row.

    Raises:
        typer.BadParameter: If the optional ``export`` extra (pandas) is not installed.
    """
    try:
        from nfield.export import results_to_dataframe
    except ImportError as exc:  # pragma: no cover - exercised only without pandas
        raise typer.BadParameter(
            "CSV output needs the export extra: pip install 'nfield[cli,export]'"
        ) from exc
    csv_text: str = results_to_dataframe(results).to_csv(index=False)
    return csv_text


def _emit_output(
    results: list[ExtractionResult],
    output: Path | None,
    fmt: OutputFormat,
    *,
    as_array: bool = False,
) -> None:
    """Serialize results in *fmt* and write them to *output* (or stdout).

    JSONL emits one compact object per result and CSV one row per result, always in
    input order. JSON emits the bare data object for a single ``extract`` result, or a
    data-object array when *as_array* is set (``batch``, whatever the document count).

    Args:
        results: The results to write (one for ``extract``, many for ``batch``).
        output: Destination file, or ``None`` to write to stdout.
        fmt: The serialization format.
        as_array: Force JSON to render an array even for a single result (batch mode).

    Raises:
        typer.BadParameter: If CSV is requested without pandas installed.
        typer.Exit: If the destination file cannot be written.
    """
    if fmt is OutputFormat.CSV:
        payload = _result_csv_text(results)
    elif fmt is OutputFormat.JSONL:
        payload = "\n".join(json.dumps(r.to_dict(), ensure_ascii=False) for r in results)
    elif as_array or len(results) != 1:
        payload = json.dumps([r.data for r in results], indent=2, ensure_ascii=False)
    else:
        payload = json.dumps(results[0].data, indent=2, ensure_ascii=False)

    if output is None:
        typer.echo(payload)
        return
    try:
        output.write_text(payload + "\n", encoding="utf-8")
    except OSError as exc:
        typer.echo(f"Could not write output to {output}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _emit_metadata(result: ExtractionResult, *, label: str | None = None) -> None:
    """Print a compact run summary for one result to stderr.

    Kept on stderr so it never pollutes piped stdout output. Grounding and
    closed-book lines appear only when those modes produced numbers.

    Args:
        result: The result to summarise.
        label: Optional prefix (e.g. a source filename) for batch runs.
    """
    meta = result.metadata
    prefix = f"[{label}] " if label else ""
    typer.echo(
        f"{prefix}status={result.status.value} quality={meta.quality_score:.3f} "
        f"K={meta.K}/{meta.K_min} (gap {meta.optimality_gap:.2f})",
        err=True,
    )
    typer.echo(
        f"{prefix}fields: {meta.fields_extracted}/{meta.fields_total} extracted, "
        f"{meta.fields_missing} missing, {meta.fields_call_failed} call-failed, "
        f"{meta.retry_rounds} retry rounds",
        err=True,
    )
    if meta.hallucination_rate is not None:
        typer.echo(
            f"{prefix}grounding: {meta.fields_grounded} grounded / "
            f"{meta.fields_ungrounded} ungrounded "
            f"(hallucination_rate {meta.hallucination_rate:.1%})",
            err=True,
        )
    if meta.answer_rate is not None and meta.abstain_rate is not None:
        typer.echo(
            f"{prefix}closed-book: answered {meta.answer_rate:.1%}, "
            f"abstained {meta.abstain_rate:.1%}",
            err=True,
        )


def _exit_on_call_failure(results: list[ExtractionResult]) -> None:
    """Exit non-zero when a field was left unextracted by an API/call failure.

    The pipeline turns a transient or hard call failure (429 / 5xx / timeout /
    model-not-found) into ``fields_call_failed`` rather than an exception, so the run
    returns a result that is real but incomplete. Surfacing it as a non-zero exit lets a
    calling script tell an incomplete run apart from a document that genuinely held
    nothing to extract. Output has already been written, so no data is lost.

    Args:
        results: The results just emitted (one for ``extract``, many for ``batch``).

    Raises:
        typer.Exit: With code 1 if any result has ``fields_call_failed > 0``.
    """
    failed_fields = sum(r.metadata.fields_call_failed for r in results)
    if failed_fields:
        failed_docs = sum(1 for r in results if r.metadata.fields_call_failed)
        reason = next((r.metadata.error for r in results if r.metadata.error), None)
        detail = f" Cause: {reason}" if reason else ""
        typer.echo(
            f"Warning: {failed_fields} field(s) across {failed_docs} document(s) were left "
            f"unextracted by API/call failures; the result is incomplete.{detail}",
            err=True,
        )
        raise typer.Exit(code=1)


def _collect_documents(inputs: list[Path], pattern: str) -> list[Path]:
    """Expand batch inputs into a sorted list of document files.

    A directory input contributes every file matching *pattern* (sorted for a
    stable order); a file input is taken as-is.

    Args:
        inputs: The paths given on the command line (files and/or directories).
        pattern: Glob applied inside each directory input.

    Returns:
        The resolved document files, directories expanded, in a deterministic order.

    Raises:
        typer.BadParameter: If an input does not exist, or a directory matches no file.
    """
    files: list[Path] = []
    for item in inputs:
        if not item.exists():
            raise typer.BadParameter(f"Input path not found: {item}")
        if item.is_dir():
            matched = sorted(p for p in item.glob(pattern) if p.is_file())
            if not matched:
                raise typer.BadParameter(f"No files matching {pattern!r} in directory: {item}")
            files.extend(matched)
        else:
            files.append(item)
    return files


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def extract(
    document: Annotated[Path, typer.Argument(help="Path to the source document (text).")],
    schema: Annotated[
        Path,
        typer.Option(
            "--schema", "-s", help="Path to a JSON Schema file.", rich_help_panel=_PANEL_CONNECTION
        ),
    ],
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model string, e.g. groq/llama-3.1-8b-instant. "
            "Falls back to $NFIELD_MODEL when omitted.",
            rich_help_panel=_PANEL_CONNECTION,
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help="Provider API key. Prefer the provider's env var; pass only for vault use.",
            rich_help_panel=_PANEL_CONNECTION,
        ),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help="Override the provider API base URL (proxy / gateway / self-hosted).",
            rich_help_panel=_PANEL_CONNECTION,
        ),
    ] = None,
    context_window: Annotated[
        int | None,
        typer.Option(
            "--context-window",
            help="Model's real context window (tokens).",
            rich_help_panel=_PANEL_CONNECTION,
        ),
    ] = None,
    max_output_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-output-tokens",
            help="Model's real max output tokens.",
            rich_help_panel=_PANEL_CONNECTION,
        ),
    ] = None,
    instructions: Annotated[
        str,
        typer.Option(
            "--instructions",
            help="Extra steering for the model, prepended.",
            rich_help_panel=_PANEL_CONNECTION,
        ),
    ] = "",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Write here instead of stdout.",
            rich_help_panel=_PANEL_OUTPUT,
        ),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            "-f",
            help="Output format: json (data), jsonl (full result), or csv.",
            rich_help_panel=_PANEL_OUTPUT,
        ),
    ] = OutputFormat.JSON,
    show_metadata: Annotated[
        bool,
        typer.Option(
            "--show-metadata",
            help="Print a run summary (status, quality, K, grounding) to stderr.",
            rich_help_panel=_PANEL_OUTPUT,
        ),
    ] = False,
    max_retry_rounds: Annotated[
        int | None,
        typer.Option(
            "--max-retry-rounds", help="Validation retry rounds.", rich_help_panel=_PANEL_TUNING
        ),
    ] = None,
    context_utilization_ratio: Annotated[
        float | None,
        typer.Option(
            "--context-utilization-ratio",
            help="Fraction of the context window used for chunks (0, 1].",
            rich_help_panel=_PANEL_TUNING,
        ),
    ] = None,
    z_target: Annotated[
        float | None,
        typer.Option(
            "--z-target", help="Output-reservation z-score.", rich_help_panel=_PANEL_TUNING
        ),
    ] = None,
    document_language: Annotated[
        str | None,
        typer.Option(
            "--document-language",
            help="BCP-47 language tag of the document (sizes the token budget).",
            rich_help_panel=_PANEL_TUNING,
        ),
    ] = None,
    chars_per_token: Annotated[
        float | None,
        typer.Option(
            "--chars-per-token",
            help="Pin the characters-per-token ratio (else script-aware estimate).",
            rich_help_panel=_PANEL_TUNING,
        ),
    ] = None,
    evidence_score_threshold: Annotated[
        float | None,
        typer.Option(
            "--evidence-score-threshold",
            help="Minimum evidence score for a chunk to enter context.",
            rich_help_panel=_PANEL_TUNING,
        ),
    ] = None,
    max_fields_per_call: Annotated[
        int | None,
        typer.Option(
            "--max-fields-per-call",
            help="Per-leaf reliability budget in difficulty-weighted units.",
            rich_help_panel=_PANEL_TUNING,
        ),
    ] = None,
    max_concurrent_calls: Annotated[
        int | None,
        typer.Option(
            "--max-concurrent-calls",
            help="Max leaf extraction calls in flight at once.",
            rich_help_panel=_PANEL_TUNING,
        ),
    ] = None,
    confidence: Annotated[
        list[str] | None,
        typer.Option(
            "--confidence",
            help="Confidence threshold TIER=SCORE (repeatable), e.g. --confidence HIGH=0.9.",
            rich_help_panel=_PANEL_TUNING,
        ),
    ] = None,
    think_budget_min: Annotated[
        int | None,
        typer.Option(
            "--think-budget-min",
            help="Lower bound of the thinking-phase token budget (set with the max).",
            rich_help_panel=_PANEL_TUNING,
        ),
    ] = None,
    think_budget_max: Annotated[
        int | None,
        typer.Option(
            "--think-budget-max",
            help="Upper bound of the thinking-phase token budget (set with the min).",
            rich_help_panel=_PANEL_TUNING,
        ),
    ] = None,
    use_advanced_sfr: Annotated[
        bool | None,
        typer.Option(
            "--advanced-sfr/--no-advanced-sfr",
            help="Enable advanced Semantic Field Routing for large schemas.",
            rich_help_panel=_PANEL_TUNING,
        ),
    ] = None,
    inject_dependencies: Annotated[
        bool | None,
        typer.Option(
            "--inject-dependencies/--no-inject-dependencies",
            help="Feed resolved dependency values into dependent leaves.",
            rich_help_panel=_PANEL_TUNING,
        ),
    ] = None,
    cascade_dependency_invalidation: Annotated[
        bool | None,
        typer.Option(
            "--cascade-invalidation/--no-cascade-invalidation",
            help="Flag dependents for revalidation when an upstream value changes.",
            rich_help_panel=_PANEL_TUNING,
        ),
    ] = None,
    strict_validation: Annotated[
        bool | None,
        typer.Option(
            "--strict-validation/--no-strict-validation",
            help="Validate values exactly as extracted (no lenient coercion).",
            rich_help_panel=_PANEL_TUNING,
        ),
    ] = None,
    reasoning_model: Annotated[
        bool | None,
        typer.Option(
            "--reasoning-model/--no-reasoning-model",
            help="Treat the model as a reasoning model and disable its thinking.",
            rich_help_panel=_PANEL_TUNING,
        ),
    ] = None,
    ground_values: Annotated[
        bool | None,
        typer.Option(
            "--ground-values/--no-ground-values",
            help="Label each value's support against the source (reports hallucination_rate).",
            rich_help_panel=_PANEL_GROUNDING,
        ),
    ] = None,
    grounding_min_score: Annotated[
        float | None,
        typer.Option(
            "--grounding-min-score",
            help="Minimum grounding score in [0, 1] to count as supported.",
            rich_help_panel=_PANEL_GROUNDING,
        ),
    ] = None,
    provenance: Annotated[
        bool | None,
        typer.Option(
            "--provenance/--no-provenance",
            help="Attach source char offsets [start, end) per value to the result.",
            rich_help_panel=_PANEL_GROUNDING,
        ),
    ] = None,
    validate_schema: Annotated[
        bool | None,
        typer.Option(
            "--validate-schema/--no-validate-schema",
            help="Reject a provably-unsatisfiable schema before any API call.",
            rich_help_panel=_PANEL_RECOVERY,
        ),
    ] = None,
    max_api_retries: Annotated[
        int | None,
        typer.Option(
            "--max-api-retries",
            help="Per-call transient-failure (429 / 5xx / timeout) retry budget.",
            rich_help_panel=_PANEL_RECOVERY,
        ),
    ] = None,
    recover_conflicts: Annotated[
        bool | None,
        typer.Option(
            "--recover-conflicts/--no-recover-conflicts",
            help="Re-extract conflicting fields during the recovery pass.",
            rich_help_panel=_PANEL_RECOVERY,
        ),
    ] = None,
    recover_call_failed: Annotated[
        bool | None,
        typer.Option(
            "--recover-call-failed/--no-recover-call-failed",
            help="Retry transiently-failed fields during the recovery pass.",
            rich_help_panel=_PANEL_RECOVERY,
        ),
    ] = None,
    fallback_model: Annotated[
        str | None,
        typer.Option(
            "--fallback-model",
            help="Stronger model to escalate still-failing fields to after recovery.",
            rich_help_panel=_PANEL_RECOVERY,
        ),
    ] = None,
    knowledge_fallback: Annotated[
        bool | None,
        typer.Option(
            "--knowledge-fallback/--no-knowledge-fallback",
            help="Fill fields absent from the document from the model's own knowledge.",
            rich_help_panel=_PANEL_KNOWLEDGE,
        ),
    ] = None,
    closed_book: Annotated[
        bool | None,
        typer.Option(
            "--closed-book/--no-closed-book",
            help="Fill the schema from model knowledge with no document (pass an empty document).",
            rich_help_panel=_PANEL_KNOWLEDGE,
        ),
    ] = None,
    self_consistency: Annotated[
        bool | None,
        typer.Option(
            "--self-consistency/--no-self-consistency",
            help="Sample each closed-book leaf twice; keep a value only if both agree.",
            rich_help_panel=_PANEL_KNOWLEDGE,
        ),
    ] = None,
) -> None:
    """Extract structured fields from a document into the given schema.

    Reads the document and schema, runs the pipeline against *model*, and writes the
    result to stdout (or ``--output``) as JSON, JSON Lines, or CSV. Every flag under
    the tuning / grounding / recovery / closed-book panels maps to an
    ``ExtractionConfig`` setting; unset flags inherit the library defaults.

    In closed-book mode there is no document to read: pass an empty file (or ``/dev/null``)
    as the document argument.
    """
    schema_dict = _load_schema(schema)
    config = _build_config(
        context_utilization_ratio=context_utilization_ratio,
        max_retry_rounds=max_retry_rounds,
        z_target=z_target,
        confidence=confidence,
        document_language=document_language,
        reasoning_model=reasoning_model,
        chars_per_token=chars_per_token,
        think_budget_min=think_budget_min,
        think_budget_max=think_budget_max,
        evidence_score_threshold=evidence_score_threshold,
        use_advanced_sfr=use_advanced_sfr,
        inject_dependencies=inject_dependencies,
        cascade_dependency_invalidation=cascade_dependency_invalidation,
        knowledge_fallback=knowledge_fallback,
        max_fields_per_call=max_fields_per_call,
        max_concurrent_calls=max_concurrent_calls,
        max_api_retries=max_api_retries,
        strict_validation=strict_validation,
        recover_conflicts=recover_conflicts,
        recover_call_failed=recover_call_failed,
        validate_schema=validate_schema,
        ground_values=ground_values,
        grounding_min_score=grounding_min_score,
        provenance=provenance,
        fallback_model=fallback_model,
        closed_book=closed_book,
        self_consistency=self_consistency,
    )
    # Closed-book ignores the document; every other mode requires one.
    document_text = "" if config.closed_book else _read_text_file(document, "Document")

    try:
        engine = NField(
            model,
            schema_dict,
            config=config,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            api_key=api_key,
            base_url=base_url,
            instructions=instructions,
        )
        result = engine.extract(document_text)
    except NFieldError as exc:
        typer.echo(f"Extraction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _emit_output([result], output, output_format)
    if output is not None:
        typer.echo(f"Wrote {result.metadata.fields_extracted} fields to {output}", err=True)
    if show_metadata:
        _emit_metadata(result)
    _exit_on_call_failure([result])


@app.command()
def batch(
    inputs: Annotated[
        list[Path],
        typer.Argument(help="Document files, and/or directories to scan for documents."),
    ],
    schema: Annotated[Path, typer.Option("--schema", "-s", help="Path to a JSON Schema file.")],
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model string. Falls back to $NFIELD_MODEL when omitted.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write results here instead of stdout."),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format (jsonl streams best for batches)."),
    ] = OutputFormat.JSONL,
    pattern: Annotated[
        str,
        typer.Option("--pattern", help="Glob applied inside directory inputs."),
    ] = _DEFAULT_BATCH_PATTERN,
    max_concurrent: Annotated[
        int | None,
        typer.Option("--max-concurrent", help="Max documents extracted in parallel."),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="Provider API key (prefer the provider env var)."),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="Override the provider API base URL."),
    ] = None,
    context_window: Annotated[
        int | None,
        typer.Option("--context-window", help="Model's real context window (tokens)."),
    ] = None,
    max_output_tokens: Annotated[
        int | None,
        typer.Option("--max-output-tokens", help="Model's real max output tokens."),
    ] = None,
    instructions: Annotated[
        str,
        typer.Option("--instructions", help="Extra steering for the model, prepended."),
    ] = "",
    max_retry_rounds: Annotated[
        int | None,
        typer.Option("--max-retry-rounds", help="Validation retry rounds."),
    ] = None,
    reasoning_model: Annotated[
        bool | None,
        typer.Option(
            "--reasoning-model/--no-reasoning-model",
            help="Treat the model as a reasoning model and disable its thinking.",
        ),
    ] = None,
    ground_values: Annotated[
        bool | None,
        typer.Option(
            "--ground-values/--no-ground-values",
            help="Label each value's support against the source.",
        ),
    ] = None,
    provenance: Annotated[
        bool | None,
        typer.Option(
            "--provenance/--no-provenance",
            help="Attach source char offsets per value.",
        ),
    ] = None,
    knowledge_fallback: Annotated[
        bool | None,
        typer.Option(
            "--knowledge-fallback/--no-knowledge-fallback",
            help="Fill absent fields from the model's own knowledge.",
        ),
    ] = None,
    fallback_model: Annotated[
        str | None,
        typer.Option("--fallback-model", help="Stronger model to escalate stragglers to."),
    ] = None,
    show_metadata: Annotated[
        bool,
        typer.Option("--show-metadata", help="Print a per-document run summary to stderr."),
    ] = False,
) -> None:
    """Extract every document in *inputs* with one reused, calibrated engine.

    File inputs are taken as-is; directory inputs contribute every file matching
    ``--pattern``. Results stream to JSON Lines (one per line, in input order) by
    default - the format that round-trips with :func:`nfield.load_results`.

    For per-field tuning (grounding thresholds, recovery, closed-book, ...), extract
    documents individually with ``nfield extract``; batch exposes the common flags.
    """
    schema_dict = _load_schema(schema)
    files = _collect_documents(inputs, pattern)
    documents = [_read_text_file(path, "Document") for path in files]

    config = _build_config(
        max_retry_rounds=max_retry_rounds,
        reasoning_model=reasoning_model,
        ground_values=ground_values,
        provenance=provenance,
        knowledge_fallback=knowledge_fallback,
        fallback_model=fallback_model,
    )
    try:
        engine = NField(
            model,
            schema_dict,
            config=config,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            api_key=api_key,
            base_url=base_url,
            instructions=instructions,
        )
        results = engine.extract_batch(documents, max_concurrent=max_concurrent)
    except NFieldError as exc:
        typer.echo(f"Batch extraction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if output is not None and output_format is OutputFormat.JSONL:
        # save_results creates parent dirs and streams the canonical JSON Lines form.
        save_results(results, output)
        typer.echo(f"Wrote {len(results)} results to {output}", err=True)
    else:
        _emit_output(results, output, output_format, as_array=True)
        if output is not None:
            typer.echo(f"Wrote {len(results)} results to {output}", err=True)

    if show_metadata:
        for path, result in zip(files, results, strict=True):
            _emit_metadata(result, label=path.name)
    _exit_on_call_failure(results)


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
