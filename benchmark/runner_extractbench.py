"""ExtractBench runner - nfield on the ContextualAI/extract-bench PDF-to-JSON tasks.

Separate from :mod:`benchmark.runner` (orchestration-layer baselines on the
in-house fixtures): this reads the cloned ExtractBench dataset under
``benchmark/external/extract-bench/``, turns each task's PDF into text through
:mod:`benchmark.pdf_router` (text layer, OCR fallback for scans), flattens its
human gold JSON, and runs nfield under the shared budget.
Results land in the standard layout:
``results/<model>_extractbench_<stamp>/<budget>/{raw,scored}`` + MANIFEST + summary.csv.

ExtractBench reports every frontier model at 0% on the 369-field ``10kq`` schema
(arXiv:2602.12247); nfield decomposes the schema so no single call approaches the
output-token wall. Only nfield runs here for now; competitor factories can be added
to :data:`ADAPTERS` once each passes a live smoke test on the shared model.

    uv run python -m benchmark.runner_extractbench --domain finance --schema credit_agreement --doc expel
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from . import report
from .adapters.nfield_adapter import NfieldAdapter
from .budget import BUDGET_MODES, resolve_budget
from .datasets import LoadedDataset
from .runner import _load_env, _now_stamp, result_dir, run_sweep
from .score import _flatten

if TYPE_CHECKING:
    from collections.abc import Callable

    from .adapters import Adapter

__all__ = ["ADAPTERS", "iter_tasks", "load_task", "main"]

_MODEL = "groq/llama-3.3-70b-versatile"
_SEEDS = 1
_DATASET = Path(__file__).resolve().parent / "external" / "extract-bench" / "dataset"
_SCHEMA_SUFFIX = "-schema.json"
_GOLD_SUFFIX = ".gold.json"

# nfield only for now; each competitor factory is added once its adapter passes a
# live smoke test on the shared model (same pattern as benchmark.runner3).
ADAPTERS: dict[str, Callable[[], Adapter]] = {"nfield": NfieldAdapter}

# Domain-agnostic faithfulness guidance, given identically to every method (the
# fairness rule). ExtractBench is a zero-shot extraction task; this is general
# instruction a real caller would write, never a gold answer.
_FAITHFULNESS = (
    "Extract each field's value exactly as written in the source document - keep all "
    "amounts, units, dates, codes, parties, and identifiers; never summarize or infer. "
    "Leave a field null if the document does not state it."
)


def iter_tasks(
    domain: str,
    schema: str,
    *,
    doc: str = "",
    limit: int = 0,
) -> list[tuple[str, Path, Path, Path]]:
    """Discover ExtractBench tasks for one ``domain/schema`` group.

    Args:
        domain: Top-level dataset domain (``finance``, ``academic``, ``hiring``,
            ``sport``).
        schema: Schema folder within the domain (e.g. ``credit_agreement``, ``10kq``).
        doc: Case-insensitive substring filter on the PDF stem; empty matches all.
        limit: Maximum number of tasks to return; ``0`` means no limit.

    Returns:
        One ``(task_name, schema_path, pdf_path, gold_path)`` tuple per PDF that has
        a sibling gold file, sorted by PDF name.

    Raises:
        FileNotFoundError: If the schema folder or its schema file is absent.
    """
    base = _DATASET / domain / schema
    schema_path = base / f"{schema}{_SCHEMA_SUFFIX}"
    if not schema_path.exists():
        raise FileNotFoundError(f"no schema at {schema_path}; is extract-bench cloned?")
    needle = doc.lower()
    tasks: list[tuple[str, Path, Path, Path]] = []
    for pdf in sorted((base / "pdf+gold").glob("*.pdf")):
        if needle and needle not in pdf.name.lower():
            continue
        gold = pdf.with_suffix("").with_suffix(_GOLD_SUFFIX)
        if not gold.exists():
            continue
        tasks.append((f"{schema}_{pdf.stem}", schema_path, pdf, gold))
    return tasks[:limit] if limit else tasks


def load_task(
    name: str,
    schema_path: Path,
    pdf_path: Path,
    gold_path: Path,
    *,
    instructions: str,
) -> LoadedDataset:
    """Read one task into a :class:`LoadedDataset` (schema + pypdfium2 text + flat gold).

    Args:
        name: Registry-style task name used in result paths.
        schema_path: The task's JSON Schema file.
        pdf_path: The source PDF; born-digital, read via its embedded text layer.
        gold_path: The human gold JSON (nested), flattened to a dot-notation key.
        instructions: Domain guidance given identically to every method.

    Returns:
        The loaded task ready for :func:`benchmark.runner.run_sweep`.
    """
    import json

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    gold = _flatten(json.loads(gold_path.read_text(encoding="utf-8")))
    return LoadedDataset(
        name=name,
        schema=schema,
        document=_pdf_text(pdf_path),
        gold=gold,
        instructions=instructions,
    )


def _pdf_text(pdf_path: Path) -> str:
    """Extract PDF text via the router: pypdfium2 text layer, OCR fallback if scanned.

    Born-digital PDFs are read exactly by pypdfium2 (BSD/Apache, no OCR); scanned PDFs
    with no text layer fall back to OCR. See :mod:`benchmark.pdf_router`.
    """
    from benchmark.pdf_router import extract

    return extract(pdf_path)


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m benchmark.runner_extractbench`` (costs API calls)."""
    _load_env()
    parser = argparse.ArgumentParser(prog="benchmark.runner_extractbench", description=__doc__)
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--domain", default="finance")
    parser.add_argument("--schema", default="credit_agreement")
    parser.add_argument("--doc", default="", help="case-insensitive PDF-name filter; empty = all")
    parser.add_argument("--limit", type=int, default=0, help="cap number of docs; 0 = all")
    parser.add_argument("--method", default="nfield", choices=sorted(ADAPTERS))
    parser.add_argument("--seeds", type=int, default=_SEEDS)
    parser.add_argument("--budget", default=BUDGET_MODES[0], choices=BUDGET_MODES)
    parser.add_argument(
        "--reasoning-model",
        action="store_true",
        help="treat the model as a reasoning model (suppress + strip thinking); set for qwen3*",
    )
    parser.add_argument("--date", default=None, help="override the result-dir stamp")
    args = parser.parse_args(argv)

    tasks = iter_tasks(args.domain, args.schema, doc=args.doc, limit=args.limit)
    if not tasks:
        raise SystemExit(
            f"no tasks for {args.domain}/{args.schema}"
            + (f" matching {args.doc!r}" if args.doc else "")
        )

    # nfield carries the reasoning-model flag; other adapters take no args.
    adapter = (
        NfieldAdapter(reasoning_model=args.reasoning_model)
        if args.method == "nfield"
        else ADAPTERS[args.method]()
    )
    limits = resolve_budget(args.budget, args.model)
    stamp = args.date or f"extractbench_{_now_stamp()}"
    run_root = result_dir(args.model, stamp)

    for name, schema_path, pdf_path, gold_path in tasks:
        dataset = load_task(name, schema_path, pdf_path, gold_path, instructions=_FAITHFULNESS)
        print(
            f"  [{args.budget}] {args.method} x {name} "
            f"({len(dataset.document):,} chars, {len(dataset.gold or {})} gold) ...",
            flush=True,
        )
        run_sweep(
            adapter,
            dataset,
            model=args.model,
            seeds=args.seeds,
            out_dir=run_root / args.budget,
            context_window=limits.context_window,
            max_output_tokens=limits.max_output_tokens,
            budget=args.budget,
            manifest_dir=run_root,
        )

    rows = report.collect_rows(run_root)
    report.write_summary_csv(rows, run_root / "summary.csv")
    print("\n" + report.format_table(rows))
    print(f"\nresults -> {run_root}")


if __name__ == "__main__":
    main()
