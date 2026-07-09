"""Flagship wide XBRL extraction: NField vs a single call on FinTagging FinNI facts.

FinTagging (Wang et al., arXiv:2505.20650) is the real XBRL-tagging benchmark: pull
every financial numeric fact out of a filing and assign its XBRL data type (one of five
item types). Its FinNI subtask scores one table at a time, and a single table fits one
model call, so a single call and NField tie there (both clear the paper's 0.7238 F1
frontier). The gap only opens when the answer is too long for one response.

This builds that regime honestly. It concatenates real FinNI filing tables into one
document and asks for every fact at once - the filing-level task a real XBRL pipeline
faces - growing the answer from ~300 to ~2400 facts. This is a wider variant of the
paper's per-table task, not the paper's exact setup: the tables, gold facts, and metric
are the benchmark's, the concatenation is ours, and it is labeled as such.

* **NField** decomposes the array and continues it across calls, so the whole answer
  comes back as the fact count climbs.
* **Single-call baseline** (the paper's own prompt, one JSON call) truncates once the
  array overruns the output budget, dropping the tail - recall falls with size.

Scored by the paper's pair-level metric: a predicted ``(fact, type)`` counts only if it
exactly matches a gold pair, as a multiset (Precision / Recall / F1). Concept linking
(FinCL) is out of scope - it maps an entity to one of 10k+ US-GAAP concepts, a taxonomy
classification, not the wide extraction measured here.

    uv run python -m benchmark.benchmarks.fintagging --sizes 1,3,6
    uv run python -m benchmark.benchmarks.fintagging               # 1,3,6,10,15 tables
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..adapters import _common

_RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"

__all__ = [
    "FINNI_TYPES",
    "WideComparison",
    "WideDocument",
    "build_wide_documents",
    "finni_f1",
    "load_finni_slice",
    "normalize_fact",
    "run_baseline",
    "run_nfield",
]

# Same model and budget across both methods (fairness: identical single-call budget).
_MODEL = "groq/qwen/qwen3.6-27b"
_CONTEXT_WINDOW = 131_000
_MAX_OUTPUT_TOKENS = 24_000

_FIXTURE = Path(__file__).resolve().parent.parent / "datasets" / "fintagging" / "finni_wide.jsonl"

# The five XBRL numeric item types FinNI classifies each fact into.
FINNI_TYPES: tuple[str, ...] = (
    "monetaryItemType",
    "perShareItemType",
    "sharesItemType",
    "percentItemType",
    "integerItemType",
)

# NField target schema: an open array of typed facts.
_FINNI_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "result": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Fact": {"type": "string", "description": "The numeric value as written."},
                    "Type": {"type": "string", "enum": list(FINNI_TYPES)},
                },
            },
        }
    },
}

# Characters stripped before comparing two facts: currency, grouping, percent, the
# parenthetical-negative wrapper and its sign (gold drops the sign on '( 213 )' -> '213').
_FACT_STRIP = re.compile(r"[\s$,%()]")

# The default document sizes (in concatenated tables), chosen to span from one table
# (fits a single call) up to the whole slice (well past a single call's output).
_DEFAULT_SIZES: tuple[int, ...] = (1, 3, 6, 10, 15)


@dataclass(frozen=True, slots=True)
class WideDocument:
    """A filing-level document built by concatenating FinNI tables.

    Args:
        n_tables: Number of source tables concatenated.
        text: The concatenated document text (fed to both methods).
        query: The paper's FinNI prompt over this document (fed to the baseline).
        instructions: The paper's instruction block (fed to NField, same guidance).
        gold: The multiset union of every table's gold facts.
    """

    n_tables: int
    text: str
    query: str
    instructions: str
    gold: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class WideComparison:
    """One document's two-method result.

    Args:
        n_tables: Tables concatenated into the document.
        gold_facts: Total gold facts to recover.
        nfield_precision, nfield_recall, nfield_f1, nfield_returned: NField scores.
        baseline_precision, baseline_recall, baseline_f1, baseline_returned: baseline.
        baseline_json_valid: Whether the baseline returned parseable JSON.
    """

    n_tables: int
    gold_facts: int
    gold_distinct: int
    nfield_precision: float
    nfield_recall: float
    nfield_f1: float
    nfield_precision_distinct: float
    nfield_recall_distinct: float
    nfield_f1_distinct: float
    nfield_returned: int
    nfield_calls: int
    baseline_precision: float
    baseline_recall: float
    baseline_f1: float
    baseline_returned: int
    baseline_json_valid: bool


def load_finni_slice(path: Path = _FIXTURE) -> list[dict[str, Any]]:
    """Read the committed FinNI wide-context slice (one JSON row per line)."""
    if not path.exists():
        raise FileNotFoundError(f"missing FinNI slice at {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalize_fact(value: Any) -> str:
    """Normalize a fact string for comparison (strip currency/grouping/sign wrappers)."""
    text = _FACT_STRIP.sub("", str(value)).strip()
    return text.lstrip("+-")


def _pairs(facts: list[dict[str, Any]]) -> Counter[tuple[str, str]]:
    # A fact contributes a (normalized value, type) pair; unclassifiable rows are skipped.
    pairs: Counter[tuple[str, str]] = Counter()
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        value = normalize_fact(fact.get("Fact", ""))
        kind = str(fact.get("Type", ""))
        if value:
            pairs[(value, kind)] += 1
    return pairs


def finni_f1(
    predicted: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> tuple[float, float, float]:
    """Return pair-level ``(precision, recall, f1)`` for predicted vs gold facts.

    A predicted ``(fact, type)`` counts as a hit only if it exactly matches a gold pair,
    matched as a multiset (repeated facts each need their own match) - the FinNI metric.
    """
    pred, want = _pairs(predicted), _pairs(gold)
    matched = sum((pred & want).values())
    n_pred, n_gold = sum(pred.values()), sum(want.values())
    return _prf(matched, n_pred, n_gold)


def finni_f1_distinct(
    predicted: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> tuple[float, float, float]:
    """Return set-level ``(precision, recall, f1)`` over DISTINCT ``(fact, type)`` pairs.

    The conservative reading of the pair metric: a repeated fact counts once. It
    reports coverage of the distinct facts a filing states, unaffected by how the
    cross-window merge dedupes exact-duplicate rows.
    """
    pred, want = set(_pairs(predicted)), set(_pairs(gold))
    return _prf(len(pred & want), len(pred), len(want))


def _prf(matched: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    precision = matched / n_pred if n_pred else 0.0
    recall = matched / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _instruction_block(query: str) -> str:
    """Return the paper's FinNI instruction text (everything before the Input marker)."""
    return query.split("Input:", 1)[0].rstrip()


def build_wide_documents(
    rows: list[dict[str, Any]], sizes: tuple[int, ...] = _DEFAULT_SIZES
) -> list[WideDocument]:
    """Concatenate the first ``n`` tables into one document, for each ``n`` in *sizes*.

    Args:
        rows: The FinNI slice rows (each a table with its context, query, and gold).
        sizes: Table counts to build documents at (clamped to the slice length).

    Returns:
        One :class:`WideDocument` per requested size, gold being the multiset union.
    """
    prefix = _instruction_block(rows[0]["query"])
    documents: list[WideDocument] = []
    for size in sizes:
        group = rows[: min(size, len(rows))]
        text = "\n\n".join(row["context"] for row in group)
        gold: list[dict[str, Any]] = []
        for row in group:
            gold.extend(json.loads(row["answer"]).get("result", []))
        query = f"{prefix}\n\n        Input: {text}\n        Output:\n        "
        documents.append(WideDocument(len(group), text, query, prefix, gold))
    return documents


def run_nfield(context: str, instructions: str, *, model: str = _MODEL) -> list[dict[str, Any]]:
    """Extract FinNI facts from one document with NField's array pipeline."""
    from nfield import nfield
    from nfield.config import ExtractionConfig

    result = nfield(
        context,
        _FINNI_SCHEMA,
        model,
        context_window=_CONTEXT_WINDOW,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
        instructions=instructions,
        config=ExtractionConfig(max_retry_rounds=1, reasoning_model=True),
    )
    facts = result.data.get("result", [])
    return facts if isinstance(facts, list) else []


def run_baseline(query: str, *, model: str = _MODEL) -> tuple[list[dict[str, Any]], bool, str]:
    """Run the paper's FinNI prompt in one JSON-mode call.

    Returns ``(facts, json_valid, raw_text)`` - the raw response so the call can be
    saved beside NField's for audit.
    """
    client = _common.groq_client(None, None)
    try:
        response = client.chat.completions.create(
            model=_common.model_id(model),
            messages=[{"role": "user", "content": query}],
            max_tokens=_MAX_OUTPUT_TOKENS,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        data = _common.parse_json_object(raw)
    except Exception:  # a truncated or invalid array is a real single-call miss
        return [], False, ""
    facts = data.get("result", [])
    return (facts if isinstance(facts, list) else []), True, raw


@contextlib.contextmanager
def _capture_calls(raw_dir: Path) -> Any:
    """Dump every NField provider call (messages in, text out) to raw/call_NNN.json."""
    from nfield.providers._base import BaseProvider

    raw_dir.mkdir(parents=True, exist_ok=True)
    original = BaseProvider.complete
    counter = {"n": 0}

    async def wrapped(self: Any, messages: Any, *, max_tokens: int) -> str:
        counter["n"] += 1
        idx = counter["n"]
        record: dict[str, Any] = {"call": idx, "max_tokens": max_tokens, "messages": messages}
        try:
            text = await original(self, messages, max_tokens=max_tokens)
            record["response"] = text
            return text
        except Exception as exc:
            record["error"] = repr(exc)
            raise
        finally:
            (raw_dir / f"call_{idx:03d}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
            )

    BaseProvider.complete = wrapped  # type: ignore[method-assign]
    try:
        yield counter
    finally:
        BaseProvider.complete = original  # type: ignore[method-assign]


def _run_size(document: WideDocument, *, model: str, size_dir: Path) -> WideComparison:
    """Run both methods on one document; write the per-size artifacts; score both metrics."""
    raw_dir = size_dir / "raw"
    with _capture_calls(raw_dir) as counter:
        nfield_facts = run_nfield(document.text, document.instructions, model=model)
    nfield_calls = int(counter["n"])
    baseline_facts, json_valid, baseline_raw = run_baseline(document.query, model=model)

    n_p, n_r, n_f = finni_f1(nfield_facts, document.gold)
    d_p, d_r, d_f = finni_f1_distinct(nfield_facts, document.gold)
    b_p, b_r, b_f = finni_f1(baseline_facts, document.gold)
    comparison = WideComparison(
        n_tables=document.n_tables,
        gold_facts=len(document.gold),
        gold_distinct=len(set(_pairs(document.gold))),
        nfield_precision=round(n_p, 4),
        nfield_recall=round(n_r, 4),
        nfield_f1=round(n_f, 4),
        nfield_precision_distinct=round(d_p, 4),
        nfield_recall_distinct=round(d_r, 4),
        nfield_f1_distinct=round(d_f, 4),
        nfield_returned=len(nfield_facts),
        nfield_calls=nfield_calls,
        baseline_precision=round(b_p, 4),
        baseline_recall=round(b_r, 4),
        baseline_f1=round(b_f, 4),
        baseline_returned=len(baseline_facts),
        baseline_json_valid=json_valid,
    )
    _dump(size_dir / "nfield_output.json", nfield_facts)
    _dump(size_dir / "gold.json", document.gold)
    _dump(size_dir / "baseline_output.json", baseline_facts)
    (raw_dir / "baseline_call.json").write_text(
        json.dumps(
            {"query": document.query, "response": baseline_raw}, ensure_ascii=False, indent=1
        ),
        encoding="utf-8",
    )
    _dump(size_dir / "summary.json", asdict(comparison))
    return comparison


def _dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def format_markdown(rows: list[WideComparison]) -> str:
    """Render the comparison as a markdown table (both metrics)."""
    header = (
        "| Tables | Gold facts | NField F1 | NField R | NField-distinct F1 | Calls "
        "| Baseline F1 | Baseline R |\n|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    lines.extend(
        f"| {r.n_tables} | {r.gold_facts} | {r.nfield_f1:.3f} | {r.nfield_recall:.3f} "
        f"| {r.nfield_f1_distinct:.3f} | {r.nfield_calls} "
        f"| {r.baseline_f1:.3f} | {r.baseline_recall:.3f} |"
        for r in rows
    )
    return "\n".join(lines)


def plot_wide_curve(rows: list[WideComparison], path: Path) -> Path | None:
    """Plot F1 vs gold-fact count for both methods, or ``None`` if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not rows:
        return None

    from ..figures import _figstyle

    ordered = sorted(rows, key=lambda r: r.gold_facts)
    facts = [r.gold_facts for r in ordered]
    path.parent.mkdir(parents=True, exist_ok=True)

    _figstyle.apply_rcparams()
    figure, axis = plt.subplots(figsize=(9, 5.4))
    _figstyle.style_axes(axis, grid_axis="both")
    axis.plot(
        facts,
        [r.nfield_f1 for r in ordered],
        marker="o",
        markersize=7,
        linewidth=2.4,
        color=_figstyle.NFIELD,
        label="nfield",
        zorder=3,
    )
    axis.plot(
        facts,
        [r.baseline_f1 for r in ordered],
        marker="s",
        markersize=6,
        linewidth=2.0,
        linestyle="--",
        color=_figstyle.BASELINE,
        label="single call",
        zorder=3,
    )
    axis.set_xlabel("gold facts in one extraction task", fontsize=10.5)
    axis.set_ylabel("pair-level F1", fontsize=10.5)
    axis.set_ylim(0.0, 1.04)
    _figstyle.title_block(
        axis,
        "nfield holds F1 as facts scale; a single call collapses",
        "FinTagging FinNI wide, qwen3.6-27b  ·  pair-level (fact, type) F1 vs facts in one task",
    )
    axis.legend(frameon=False, fontsize=10, loc="lower left", handlelength=1.8)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


_CSV_COLUMNS: tuple[str, ...] = (
    "n_tables",
    "gold_facts",
    "gold_distinct",
    "nfield_p",
    "nfield_r",
    "nfield_f1",
    "nfield_p_distinct",
    "nfield_r_distinct",
    "nfield_f1_distinct",
    "nfield_returned",
    "nfield_calls",
    "baseline_p",
    "baseline_r",
    "baseline_f1",
    "baseline_returned",
    "baseline_json_valid",
)


def _write_csv(rows: list[WideComparison], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CSV_COLUMNS)
        for r in rows:
            writer.writerow(
                [
                    r.n_tables,
                    r.gold_facts,
                    r.gold_distinct,
                    r.nfield_precision,
                    r.nfield_recall,
                    r.nfield_f1,
                    r.nfield_precision_distinct,
                    r.nfield_recall_distinct,
                    r.nfield_f1_distinct,
                    r.nfield_returned,
                    r.nfield_calls,
                    r.baseline_precision,
                    r.baseline_recall,
                    r.baseline_f1,
                    r.baseline_returned,
                    int(r.baseline_json_valid),
                ]
            )


def _load_env() -> None:
    # A live run needs GROQ_API_KEY; mirror the sweeps and read a local .env.
    env = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _parse_sizes(spec: str) -> tuple[int, ...]:
    sizes = tuple(int(s) for s in spec.split(",") if s.strip())
    return sizes or _DEFAULT_SIZES


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m benchmark.benchmarks.fintagging``."""
    _load_env()
    parser = argparse.ArgumentParser(prog="benchmark.benchmarks.fintagging", description=__doc__)
    parser.add_argument(
        "--sizes", default=",".join(map(str, _DEFAULT_SIZES)), help="table counts per document"
    )
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    # All FinTagging runs group under one results/fintagging/ parent, each a
    # timestamped <provider>-<model>-<stamp> folder.
    if args.out is None:
        provider = args.model.split("/", 1)[0]
        model_tag = (
            args.model.split("/", 1)[1].replace("/", "-") if "/" in args.model else args.model
        )
        stamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        args.out = _RESULTS_ROOT / "fintagging" / f"{provider}-{model_tag}-{stamp}"

    documents = build_wide_documents(load_finni_slice(), _parse_sizes(args.sizes))
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[WideComparison] = []
    for document in documents:
        print(f"  {document.n_tables} tables, {len(document.gold)} gold facts ...", flush=True)
        size_dir = args.out / f"{document.n_tables}table"
        rows.append(_run_size(document, model=args.model, size_dir=size_dir))
        time.sleep(1.0)

    table = format_markdown(rows)
    _write_csv(rows, args.out / "summary.csv")
    (args.out / "summary.md").write_text(table + "\n", encoding="utf-8")
    plot_wide_curve(rows, args.out / "wide_f1_vs_facts.png")
    _write_manifest(rows, args.out, model=args.model)
    print("\n" + table)
    print(f"\nsaved -> {args.out}")


def _write_manifest(rows: list[WideComparison], out: Path, *, model: str) -> None:
    """Write the run manifest, mirroring the ExtractBench MANIFEST.json shape."""
    manifest = {
        "benchmark": "fintagging_finni_wide",
        "task_note": (
            "Wider variant of the FinNI per-table task (arXiv:2505.20650): real FinNI "
            "tables concatenated into one document; the concatenation is ours, the "
            "tables, gold, and pair-level metric are the benchmark's."
        ),
        "model": model,
        "reasoning_model": True,
        "sizes": [r.n_tables for r in rows],
        "metrics": {
            "multiset": "pair-level (fact, type) multiset P/R/F1 (the paper metric)",
            "distinct": "set-level distinct (fact, type) P/R/F1 (conservative reading)",
        },
        "results": [asdict(r) for r in rows],
        "recorded_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
