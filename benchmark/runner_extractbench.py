"""ExtractBench sweep - nfield over every dataset in the cloned extract-bench repo.

Self-contained: discovers every ``domain/schema`` group under
``benchmark/external/extract-bench/dataset``, turns each task PDF into text through
:mod:`benchmark.pdf_router` (text layer, two-engine OCR for scans), runs nfield, and
scores against the human gold with :mod:`benchmark.score`. Results use their own
layout, one folder per dataset::

    results/<provider>-extractbench-<model>-<stamp>/
      native/
        MANIFEST.json            run-level provenance
        summary.csv              aggregate row per dataset
        <dataset>/
          MANIFEST.json          dataset-level provenance
          summary.csv            row per document
          raw/<doc>.json         extracted data + run metadata
          scored/<doc>.json      coverage, value accuracy, outcome counts

Rate handling: nfield retries 429s honouring Retry-After and caps in-flight calls;
the sweep adds a short pause between documents and records a failed document as a
zero-score row instead of aborting.

    uv run python -m benchmark.runner_extractbench
    uv run python -m benchmark.runner_extractbench --datasets sport_swimming --limit 1
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .pdf_router import MIN_CHARS_PER_PAGE, extract, text_layer
from .score import _flatten
from .score_extractbench import score_extractbench

__all__ = ["discover_datasets", "main", "run_dataset"]

_MODEL = "groq/qwen/qwen3.6-27b"
_CONTEXT_WINDOW = 131_000
_MAX_OUTPUT_TOKENS = 24_000
_BUDGET = "native"
_DATASET_ROOT = Path(__file__).resolve().parent / "external" / "extract-bench" / "dataset"
_RESULTS_ROOT = Path(__file__).resolve().parent / "results"
_SCHEMA_SUFFIX = "-schema.json"
_GOLD_SUFFIX = ".gold.json"
# Pause between documents so per-minute token windows recover between large calls.
_PAUSE_SECONDS = 3.0

# Domain-agnostic faithfulness guidance given to every document identically.
_INSTRUCTIONS = (
    "Extract each field's value exactly as written in the document; keep names, "
    "numbers, dates, units, and identifiers. For arrays, include every item the "
    "document lists. Leave a field null if the document does not state it."
)
# Appended when the router used OCR: the text is one or two noisy renditions.
_OCR_INSTRUCTIONS = (
    " The document text comes from OCR of a scanned image and may contain character "
    "recognition errors; two independent OCR renditions may be given - cross-check "
    "them. When a value is clearly garbled, output the obvious intended form."
)


def discover_datasets() -> dict[str, Path]:
    """Map dataset name (``domain_schema``) to its schema folder, sorted by name."""
    found: dict[str, Path] = {}
    for schema_file in sorted(_DATASET_ROOT.rglob(f"*{_SCHEMA_SUFFIX}")):
        base = schema_file.parent
        name = f"{base.parent.name}_{base.name}"
        found[name] = base
    return found


def run_dataset(
    name: str,
    base: Path,
    out_dir: Path,
    *,
    model: str,
    reasoning_model: bool,
    doc_filter: str = "",
    limit: int = 0,
    llm_judge: bool = False,
    ground_values: bool = False,
) -> dict[str, Any]:
    """Run nfield on every document of one dataset and write raw/scored/summary.

    Args:
        name: Dataset name used in paths (e.g. ``finance_10kq``).
        base: Dataset folder holding ``<schema>-schema.json`` and ``pdf+gold/``.
        out_dir: The run's ``native/`` folder; dataset output goes under it.
        model: Provider-prefixed model id.
        reasoning_model: Suppress model reasoning tokens (thinking off).
        doc_filter: Case-insensitive substring filter on PDF names; empty runs all.
        limit: Max documents; ``0`` runs all.

    Returns:
        The aggregate summary row for this dataset.
    """
    schema_path = base / f"{base.name}{_SCHEMA_SUFFIX}"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    # A few dataset schemas wrap the JSON Schema in a descriptor object.
    if "schema_definition" in schema:
        schema = schema["schema_definition"]

    ds_dir = out_dir / name
    raw_dir, scored_dir = ds_dir / "raw", ds_dir / "scored"
    raw_dir.mkdir(parents=True, exist_ok=True)
    scored_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    pdfs = sorted((base / "pdf+gold").glob("*.pdf"))
    if doc_filter:
        pdfs = [p for p in pdfs if doc_filter.lower() in p.name.lower()]
    if limit:
        pdfs = pdfs[:limit]

    for pdf in pdfs:
        gold_path = pdf.with_suffix("").with_suffix(_GOLD_SUFFIX)
        if not gold_path.exists():
            continue
        doc_name = pdf.stem
        started = time.monotonic()
        try:
            row = _run_document(
                doc_name,
                pdf,
                gold_path,
                schema,
                raw_dir,
                scored_dir,
                model=model,
                reasoning_model=reasoning_model,
                llm_judge=llm_judge,
                ground_values=ground_values,
            )
        except Exception as exc:  # a failed document scores zero, never aborts the sweep
            row = {"document": doc_name, "status": f"error: {exc}"[:120]}
            gold = _flatten(json.loads(gold_path.read_text(encoding="utf-8")))
            row.update({"gold_fields": len(gold), "coverage": 0.0, "value_accuracy": 0.0, "K": 0})
        row["seconds"] = round(time.monotonic() - started, 1)
        rows.append(row)
        print(
            f"  {name}/{row['document']}: cov {row['coverage']:.3f}  "
            f"va {row['value_accuracy']:.3f}  K {row['K']}  {row['seconds']}s  {row['status']}"
        )
        time.sleep(_PAUSE_SECONDS)

    _write_csv(ds_dir / "summary.csv", rows)
    aggregate = _aggregate(name, rows)
    _write_json(
        ds_dir / "MANIFEST.json",
        {
            "dataset": name,
            "schema_file": str(schema_path.relative_to(_DATASET_ROOT)),
            "documents": len(rows),
            "model": model,
            "reasoning_model": reasoning_model,
            "budget": _BUDGET,
            "aggregate": aggregate,
            "recorded_at": _now_iso(),
        },
    )
    return aggregate


def _run_document(
    doc_name: str,
    pdf: Path,
    gold_path: Path,
    schema: dict[str, Any],
    raw_dir: Path,
    scored_dir: Path,
    *,
    model: str,
    reasoning_model: bool,
    llm_judge: bool = False,
    ground_values: bool = False,
) -> dict[str, Any]:
    """Extract one PDF, score it, and write its raw and scored artifacts."""
    from nfield import nfield
    from nfield.config import ExtractionConfig

    layer_text, pages = text_layer(pdf)
    scanned = not pages or len(layer_text.strip()) / pages < MIN_CHARS_PER_PAGE
    document = extract(pdf)
    instructions = _INSTRUCTIONS + (_OCR_INSTRUCTIONS if scanned else "")

    result = nfield(
        document,
        schema,
        model,
        context_window=_CONTEXT_WINDOW,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
        instructions=instructions,
        config=ExtractionConfig(
            max_retry_rounds=1,
            reasoning_model=reasoning_model,
            ground_values=ground_values,
        ),
    )
    # Write the raw extraction before scoring so a scorer failure cannot lose a
    # finished run.
    _write_json(
        raw_dir / f"{doc_name}.json",
        {
            "document": doc_name,
            "route": "ocr" if scanned else "text-layer",
            "doc_chars": len(document),
            "data": result.data,
            "metadata": {
                "K": result.metadata.K,
                "calls_by_origin": result.metadata.calls_by_origin,
                "fields_extracted": result.metadata.fields_extracted,
                "fields_total": result.metadata.fields_total,
                "fields_missing": result.metadata.fields_missing,
                "status": result.status.value,
            },
        },
    )
    report, gold = score_extractbench(
        result.data, json.loads(gold_path.read_text(encoding="utf-8")), schema
    )
    if llm_judge:
        report = _rejudge_with_llm(report, gold, schema, model, reasoning_model)
    # Every non-correct field ships with its gold and predicted value, so a scored
    # file is auditable on its own: the judgement is inspectable, not just a number.
    mismatches = [
        {
            "path": fs.path,
            "outcome": fs.outcome.value,
            "gold": fs.gold,
            "predicted": fs.predicted,
        }
        for fs in report.fields
        if fs.outcome.value != "correct"
    ]
    _write_json(
        scored_dir / f"{doc_name}.json",
        {
            "document": doc_name,
            "gold_fields": report.n_fields,
            "coverage": round(report.coverage, 4),
            "value_accuracy": round(report.value_accuracy, 4),
            "outcomes": {o.value: n for o, n in report.outcomes.items()},
            "by_type": {
                ft.value: {"correct": st.correct, "total": st.total}
                for ft, st in report.by_type.items()
            },
            "mismatches": mismatches,
        },
    )
    return {
        "document": doc_name,
        "gold_fields": report.n_fields,
        "coverage": round(report.coverage, 4),
        "value_accuracy": round(report.value_accuracy, 4),
        "K": result.metadata.K,
        "status": result.status.value,
    }


def _rejudge_with_llm(
    report: Any, gold: dict[str, Any], schema: dict[str, Any], model: str, reasoning_model: bool
) -> Any:
    """Re-judge deterministic misses under the benchmark's LLM tiers (score_extractbench.llm_rejudge)."""
    import asyncio

    from nfield.providers.groq._provider import GroqProvider

    from .score_extractbench import llm_rejudge

    judge = GroqProvider(model.split("/", 1)[1], reasoning_model=reasoning_model)

    async def complete(prompt: str) -> str:
        return await judge.complete([{"role": "user", "content": prompt}], max_tokens=2000)

    try:
        return asyncio.run(llm_rejudge(report, gold, schema, complete))
    except Exception as exc:  # the judge may only confirm, never cost a run
        print(f"  llm judge skipped: {exc}", flush=True)
        return report


def _aggregate(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Field-weighted aggregate over a dataset's document rows."""
    total_fields = sum(r["gold_fields"] for r in rows)
    if not rows or not total_fields:
        return {"dataset": name, "documents": len(rows), "coverage": 0.0, "value_accuracy": 0.0}
    weighted = lambda key: sum(r[key] * r["gold_fields"] for r in rows) / total_fields  # noqa: E731
    return {
        "dataset": name,
        "documents": len(rows),
        "gold_fields": total_fields,
        "coverage": round(weighted("coverage"), 4),
        "value_accuracy": round(weighted("value_accuracy"), 4),
        "K_total": sum(r["K"] for r in rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds")


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_env() -> None:
    import os

    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m benchmark.runner_extractbench`` (costs API calls)."""
    parser = argparse.ArgumentParser(prog="benchmark.runner_extractbench", description=__doc__)
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument(
        "--reasoning-model",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="suppress reasoning tokens (default on for the qwen default model)",
    )
    parser.add_argument("--datasets", default="", help="comma-separated names; empty = all")
    parser.add_argument("--doc", default="", help="substring filter on PDF names")
    parser.add_argument("--limit", type=int, default=0, help="max documents per dataset")
    parser.add_argument(
        "--llm-judge",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="re-judge string_semantic / array_llm misses with the model, as the "
        "official ExtractBench harness does (costs extra API calls)",
    )
    parser.add_argument(
        "--ground-values",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="enable the grounding gate: re-extract values the source text does not support",
    )
    args = parser.parse_args(argv)

    _load_env()
    provider = args.model.split("/", 1)[0]
    model_tag = args.model.split("/", 1)[1].replace("/", "-")
    stamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = _RESULTS_ROOT / f"{provider}-extractbench-{model_tag}-{stamp}" / _BUDGET
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets()
    if args.datasets:
        wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
        datasets = {k: v for k, v in datasets.items() if k in wanted}
    print(f"model {args.model}  datasets {list(datasets)}  -> {out_dir}")

    aggregates: list[dict[str, Any]] = []
    for name, base in datasets.items():
        print(f"== {name} ==")
        aggregates.append(
            run_dataset(
                name,
                base,
                out_dir,
                model=args.model,
                reasoning_model=args.reasoning_model,
                doc_filter=args.doc,
                limit=args.limit,
                llm_judge=args.llm_judge,
                ground_values=args.ground_values,
            )
        )

    _write_csv(out_dir / "summary.csv", aggregates)
    _write_json(
        out_dir / "MANIFEST.json",
        {
            "benchmark": "extract-bench",
            "model": args.model,
            "reasoning_model": args.reasoning_model,
            "budget": _BUDGET,
            "context_window": _CONTEXT_WINDOW,
            "max_output_tokens": _MAX_OUTPUT_TOKENS,
            "datasets": aggregates,
            "git_commit": _git_commit(),
            "recorded_at": _now_iso(),
        },
    )
    print(f"\nresults -> {out_dir}")
    for agg in aggregates:
        print(
            f"  {agg['dataset']:26s} docs {agg['documents']:2d}  "
            f"cov {agg['coverage']:.3f}  va {agg['value_accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
