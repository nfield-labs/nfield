"""Head-to-head: nfield vs orchestration-layer extractors on one wide document.

Runs every method on the *same* document, the *same* JSON Schema, the *same*
model, and the *same* input/output budget, then scores them all with the *one*
ExtractBench scorer so the numbers are directly comparable. The flagship task is
the ExtractBench ``finance/10kq`` filing (369 fields in the benchmark's own count;
its gold key flattens to more scored values), where the published single-call pass
rate is 0% across frontier models (ExtractBench, arXiv:2602.12247).

Fairness rules (identical to the sweep runners):
  - one shared model per run: on a reasoning model nfield strips the trace (its own
    capability) and the json-mode baselines avoid it via forced JSON, so the run
    stays same-model and the difference measured is the extraction strategy;
  - one shared context window and output budget, applied by every adapter;
  - a failed run stays in the denominator as a miss - refusing a hard schema is
    exactly the capability under test, so it is never dropped.

Run several models and draw one grouped chart with ``--combine`` to show how each
method's accuracy moves with the model, each model keeping its own comparison.

Output, one folder per run::

    results/head2head/<model>-<stamp>/
      MANIFEST.json      run provenance (model, doc, schema, commit, budget)
      summary.csv        one row per method
      summary.md         the same table, rendered
      summary.png        value-accuracy bar chart
      raw/<method>.json      extracted data + run metrics
      scored/<method>.json   value accuracy, coverage, outcome counts

    uv run python -m benchmark.benchmarks.head2head
    uv run python -m benchmark.benchmarks.head2head --judge --methods nfield,instructor
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..adapters.instructor_adapter import InstructorAdapter
from ..adapters.langchain_adapter import LangChainAdapter
from ..adapters.langstruct_adapter import LangStructAdapter
from ..adapters.native_json_adapter import NativeJsonAdapter
from ..adapters.nfield_adapter import NfieldAdapter
from ..pdf_router import extract
from ..scoring.score import _flatten
from ..scoring.score_extractbench import score_extractbench

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..adapters import Adapter, AdapterOutput
    from ..scoring.score import ScoreReport

__all__ = ["combine_runs", "load_task", "main", "regen_reports", "run_head_to_head"]

# Default shared model. Non-thinking, so the baselines (which cannot strip a
# reasoning trace) are judged on extraction alone; pass --model to run others,
# including reasoning models where nfield strips the trace and the json-mode
# baselines avoid it via forced JSON.
_MODEL = "groq/llama-3.3-70b-versatile"
_CONTEXT_WINDOW = 131_000
_MAX_OUTPUT_TOKENS = 24_000

_DATASET_ROOT = Path(__file__).resolve().parent.parent / "external" / "extract-bench" / "dataset"
_RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results" / "head2head"
# Flagship task: the 369-field 10-Q where single-call pass rate is 0% (ExtractBench).
_TASK_SCHEMA = _DATASET_ROOT / "finance" / "10kq" / "10kq-schema.json"
_TASK_PDF = _DATASET_ROOT / "finance" / "10kq" / "pdf+gold" / "nke_10q_fy2025q2.pdf"
_TASK_GOLD = _DATASET_ROOT / "finance" / "10kq" / "pdf+gold" / "nke_10q_fy2025q2.gold.json"

# nfield ships the whole filing per leaf (tens of thousands of tokens); at the
# default concurrency several such calls exceed a hosted 250K TPM window and lose
# leaves to 429 exhaustion. Two in flight stays under the ceiling for a fair run.
_NFIELD_THROTTLE: int = 2
# Cooldown between methods so the per-minute token window refills before the next.
_METHOD_COOLDOWN_S: float = 20.0

# Domain-agnostic faithfulness guidance, given identically to every method.
_INSTRUCTIONS = (
    "Extract each field's value exactly as written in the document; keep names, "
    "numbers, dates, units, and identifiers. For arrays, include every item the "
    "document lists. Leave a field null if the document does not state it."
)

# Method name -> zero-arg adapter factory. nfield plus the orchestration-layer
# baselines that run over a hosted API (the decoding-layer tools need local
# weights and are a different substrate, so they are out of this comparison).
# langextract is excluded: it has no Groq provider, and driven through its
# OpenAI-compatible path it errors on the endpoint's JSON validation rather than
# producing a comparable single-call result, so its score would not be a fair one.
ROSTER: dict[str, Callable[[], Adapter]] = {
    "nfield": NfieldAdapter,
    "native_json": NativeJsonAdapter,
    "instructor": InstructorAdapter,
    "langchain": LangChainAdapter,
    "langstruct": LangStructAdapter,
}


def _is_thinking_model(model: str) -> bool:
    """Whether the model needs its inline reasoning trace stripped (qwen3, r1, qwq).

    gpt-oss is excluded on purpose: on Groq its reasoning is returned in a separate
    ``reasoning`` field, so the content is already clean and no strip is needed.
    """
    m = model.lower()
    return "qwen3" in m or "qwq" in m or "r1" in m


def _build_adapter(name: str, model: str) -> Adapter:
    """Instantiate a method's adapter.

    nfield's reasoning strip is enabled on a thinking model - a capability only
    nfield has. The competitor libraries expose no such option, so they run
    unchanged: that difference is a real property of the tools, not a handicap
    the benchmark imposes.
    """
    if name == "nfield":
        return NfieldAdapter(
            reasoning_model=_is_thinking_model(model),
            max_concurrent_calls=_NFIELD_THROTTLE,
        )
    return ROSTER[name]()


def load_task() -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Return the flagship ``(document, schema, gold_document)`` triple.

    The PDF is turned into text once and shared by every method. The schema is
    unwrapped if it is nested in a descriptor object.

    Returns:
        The document text, the JSON Schema, and the nested gold document.
    """
    schema = json.loads(_TASK_SCHEMA.read_text(encoding="utf-8"))
    if "schema_definition" in schema:
        schema = schema["schema_definition"]
    document = extract(_TASK_PDF)
    gold_document = json.loads(_TASK_GOLD.read_text(encoding="utf-8"))
    return document, schema, gold_document


def run_head_to_head(
    methods: list[str],
    *,
    model: str,
    judge: bool,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Run each method on the flagship task, score it, and persist artifacts.

    Args:
        methods: Method names to run, each a key of :data:`ROSTER`.
        model: Provider-qualified model id shared by every method.
        judge: When True, re-score deterministic misses under the benchmark's own
            LLM tiers (the official ``string_semantic`` / ``array_llm`` judge).
        out_dir: The run directory; ``raw/`` and ``scored/`` are created under it.

    Returns:
        One summary row per method, in run order.
    """
    document, schema, gold_document = load_task()
    # The scorer keeps every flattened gold path (NOT_FOUND becomes an expected
    # absence), so the field count is the full flattened key.
    gold_fields = len(_flatten(gold_document))
    raw_dir, scored_dir = out_dir / "raw", out_dir / "scored"
    raw_dir.mkdir(parents=True, exist_ok=True)
    scored_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, name in enumerate(methods):
        if index:
            time.sleep(_METHOD_COOLDOWN_S)
        adapter = _build_adapter(name, model)
        print(f"  {name}: running on {model} ...", flush=True)
        out = adapter.run(
            document,
            schema,
            model=model,
            context_window=_CONTEXT_WINDOW,
            max_output_tokens=_MAX_OUTPUT_TOKENS,
            instructions=_INSTRUCTIONS,
        )
        row = _score_and_persist(
            name, out, gold_document, schema, raw_dir, scored_dir, model=model, judge=judge
        )
        rows.append(row)
        print(
            f"    va {row['value_accuracy']:.3f}"
            + (f"  judged {row['value_accuracy_judged']:.3f}" if judge else "")
            + f"  cov {row['coverage']:.3f}  K {row['K']}  {row['elapsed_s']}s"
            + (f"  [{row['error_category']}]" if row["error_category"] else ""),
            flush=True,
        )

    _write_csv(out_dir / "summary.csv", rows)
    _write_markdown(
        out_dir / "summary.md", rows, model=model, gold_fields=gold_fields, judge=judge
    )
    _plot_summary(out_dir / "summary.png", rows, gold_fields=gold_fields, judge=judge)
    _write_manifest(
        out_dir / "MANIFEST.json", rows, model=model, gold_fields=gold_fields, judge=judge
    )
    return rows


def _score_and_persist(
    name: str,
    out: AdapterOutput,
    gold_document: dict[str, Any],
    schema: dict[str, Any],
    raw_dir: Path,
    scored_dir: Path,
    *,
    model: str,
    judge: bool,
) -> dict[str, Any]:
    """Score one method's output and write its raw + scored files. Returns the summary row."""
    report, gold = score_extractbench(out.data, gold_document, schema, call_failed=out.call_failed)
    strict_va = report.value_accuracy
    judged_va = strict_va
    if judge and out.data:
        judged_report = _judge_report(report, gold, schema, model)
        judged_va = judged_report.value_accuracy

    _write_json(
        raw_dir / f"{name}.json",
        {
            "method": name,
            "data": out.data,
            "fields_total": out.fields_total,
            "fields_extracted": out.fields_extracted,
            "K": out.k,
            "K_min": out.k_min,
            "call_failed": out.call_failed,
            "elapsed_s": round(out.elapsed_seconds, 2),
            "error": out.error,
            "error_category": out.error_category,
        },
    )
    _write_json(
        scored_dir / f"{name}.json",
        {
            "method": name,
            "gold_fields": report.n_fields,
            "value_accuracy": round(strict_va, 4),
            "value_accuracy_judged": round(judged_va, 4),
            "coverage": round(report.coverage, 4),
            "outcomes": {o.value: n for o, n in report.outcomes.items()},
        },
    )
    return {
        "method": name,
        "value_accuracy": round(strict_va, 4),
        "value_accuracy_judged": round(judged_va, 4),
        "coverage": round(report.coverage, 4),
        "gold_fields": report.n_fields,
        "fields_extracted": out.fields_extracted,
        "K": out.k,
        "elapsed_s": round(out.elapsed_seconds, 1),
        "error_category": out.error_category or "",
    }


def _judge_report(
    report: ScoreReport, gold: dict[str, Any], schema: dict[str, Any], model: str
) -> ScoreReport:
    """Re-judge deterministic misses under the benchmark's LLM tiers (non-thinking model)."""
    import asyncio

    from nfield.providers.groq._provider import GroqProvider

    from ..scoring.score_extractbench import llm_rejudge

    judge = GroqProvider(model.split("/", 1)[1])

    async def complete(prompt: str) -> str:
        return await judge.complete([{"role": "user", "content": prompt}], max_tokens=2000)

    try:
        return asyncio.run(llm_rejudge(report, gold, schema, complete))
    except Exception as exc:  # the judge only confirms misses; it must never cost the run
        print(f"    llm judge skipped: {exc}", flush=True)
        return report


def _plot_summary(
    path: Path, rows: list[dict[str, Any]], *, gold_fields: int, judge: bool
) -> None:
    """Draw a value-accuracy bar chart, nfield highlighted, and save it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda r: r["value_accuracy"], reverse=True)
    labels = [r["method"] for r in ordered]
    strict = [r["value_accuracy"] for r in ordered]
    highlight = "#1f6feb"
    muted = "#8b949e"
    colors = [highlight if r["method"] == "nfield" else muted for r in ordered]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(labels))
    if judge:
        judged = [r["value_accuracy_judged"] for r in ordered]
        width = 0.4
        ax.bar([i - width / 2 for i in x], strict, width, color=colors, label="strict")
        ax.bar(
            [i + width / 2 for i in x],
            judged,
            width,
            color=colors,
            alpha=0.55,
            label="judged",
        )
        ax.legend(frameon=False)
    else:
        ax.bar(list(x), strict, 0.6, color=colors)
    for i, v in enumerate(strict):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Value accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        f"Wide-schema extraction: 10-Q, {gold_fields} scored gold values, same model + doc"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


_CSV_COLUMNS = (
    "method",
    "value_accuracy",
    "value_accuracy_judged",
    "coverage",
    "gold_fields",
    "fields_extracted",
    "K",
    "elapsed_s",
    "error_category",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path, rows: list[dict[str, Any]], *, model: str, gold_fields: int, judge: bool
) -> None:
    ordered = sorted(rows, key=lambda r: r["value_accuracy"], reverse=True)
    thinking = _is_thinking_model(model)
    reasoning_line = (
        f"Model `{model}`, a reasoning model: nfield strips its reasoning trace, the "
        "competitor libraries see the raw completion (json-mode baselines avoid it via "
        "forced JSON)."
        if thinking
        else f"Model `{model}`, non-thinking, so no method gains on reasoning tokens."
    )
    lines = [
        f"# Head-to-head: 10-Q, {gold_fields} scored gold values, same model, same document",
        "",
        reasoning_line,
        "Every method scored with the one ExtractBench scorer. A failed run stays a miss.",
        "",
        "| Method | Value accuracy | " + ("Judged | " if judge else "") + "Coverage | K | Fail |",
        "|---|---:|" + ("---:|" if judge else "") + "---:|---:|---|",
    ]
    for r in ordered:
        judged = f" {r['value_accuracy_judged']:.3f} |" if judge else ""
        lines.append(
            f"| {r['method']} | {r['value_accuracy']:.3f} |{judged} "
            f"{r['coverage']:.3f} | {r['K']} | {r['error_category'] or '-'} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(
    path: Path, rows: list[dict[str, Any]], *, model: str, gold_fields: int, judge: bool
) -> None:
    _write_json(
        path,
        {
            "task": "extractbench/finance_10kq/csco_10q_fy2025q2",
            "gold_fields": gold_fields,
            "model": model,
            "context_window": _CONTEXT_WINDOW,
            "max_output_tokens": _MAX_OUTPUT_TOKENS,
            "judge": judge,
            "scorer": "score_extractbench (shared by every method)",
            "reference": "ExtractBench arXiv:2602.12247 (single-call pass rate 0% at this width)",
            "git_commit": _git_commit(),
            "recorded_at": _now_iso(),
            "rows": rows,
        },
    )


def regen_reports(run_dir: Path) -> None:
    """Rebuild a run's summary table and chart from its stored MANIFEST.

    Lets a reporting fix reach an existing run without re-calling the model - the
    MANIFEST already carries every scored row.
    """
    manifest = json.loads((run_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    model = manifest["model"]
    # Keep only methods still in the roster, so a dropped method leaves the report.
    rows = [r for r in manifest["rows"] if r["method"] in ROSTER]
    gold_fields, judge = manifest["gold_fields"], manifest["judge"]
    _write_csv(run_dir / "summary.csv", rows)
    _write_markdown(
        run_dir / "summary.md", rows, model=model, gold_fields=gold_fields, judge=judge
    )
    _plot_summary(run_dir / "summary.png", rows, gold_fields=gold_fields, judge=judge)


def combine_runs(run_dirs: list[Path], out_dir: Path) -> None:
    """Draw one grouped value-accuracy chart across several single-model runs.

    Each input directory is a completed run (its ``MANIFEST.json`` names the model,
    its ``scored/`` holds one file per method). The chart groups the methods on the
    x-axis and draws one bar per model, so a method's model-dependence is visible
    and every model keeps its own same-model comparison.

    Args:
        run_dirs: Completed run directories to combine.
        out_dir: Where the grouped chart and combined table are written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    models: list[tuple[str, dict[str, float]]] = []
    for d in run_dirs:
        manifest = json.loads((d / "MANIFEST.json").read_text(encoding="utf-8"))
        scores = {
            s["method"]: s["value_accuracy"]
            for f in sorted((d / "scored").glob("*.json"))
            for s in [json.loads(f.read_text(encoding="utf-8"))]
        }
        models.append((manifest["model"], scores))

    _plot_grouped(out_dir / "summary_grouped.png", models)
    method_order = list(ROSTER)
    lines = ["method," + ",".join(m.split("/", 1)[-1] for m, _ in models)]
    for name in method_order:
        cells = [f"{scores.get(name, 0.0):.4f}" for _, scores in models]
        lines.append(name + "," + ",".join(cells))
    (out_dir / "summary_grouped.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"combined -> {out_dir / 'summary_grouped.png'}", flush=True)


def _plot_grouped(path: Path, models: list[tuple[str, dict[str, float]]]) -> None:
    """Grouped value-accuracy bars: methods on x, one bar per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..figures import _figstyle

    method_order = list(ROSTER)
    nfield_top = max((s.get("nfield", 0.0) for _, s in models), default=1.0)
    _figstyle.apply_rcparams()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    _figstyle.style_axes(ax)
    width = 0.8 / max(1, len(models))
    x = range(len(method_order))
    for mi, (model, scores) in enumerate(models):
        offset = (mi - (len(models) - 1) / 2) * width
        heights = [scores.get(name, 0.0) for name in method_order]
        bars = ax.bar(
            [i + offset for i in x],
            heights,
            width,
            color=_figstyle.MODEL_PALETTE[mi % len(_figstyle.MODEL_PALETTE)],
            label=model.split("/", 1)[-1],
            zorder=3,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color=_figstyle.TEXT_MUTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(method_order, fontsize=10)
    ax.set_ylabel("value accuracy (per gold field)", fontsize=10.5)
    ax.set_ylim(0, 1.05)
    _figstyle.title_block(
        ax,
        "nfield vs single-call libraries on a wide 10-Q",
        f"ExtractBench 10-Q, same model within each group  ·  nfield tops the field at "
        f"{nfield_top:.0%}",
    )
    ax.legend(frameon=False, title="model", fontsize=9.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _now_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds")


def _now_stamp() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_env() -> None:
    """Load ``.env`` at the repo root so GROQ_API_KEY is available, if present."""
    env = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        import os

        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> None:
    """CLI: run the head-to-head and write the result folder."""
    parser = argparse.ArgumentParser(prog="benchmark.benchmarks.head2head", description=__doc__)
    parser.add_argument("--model", default=_MODEL, help="provider-qualified model id")
    parser.add_argument("--methods", default=",".join(ROSTER), help="comma-separated method names")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="re-score misses under the benchmark's LLM tiers (extra calls)",
    )
    parser.add_argument(
        "--combine",
        default="",
        help="comma-separated completed run dirs; draw a grouped chart and exit (no calls)",
    )
    args = parser.parse_args(argv)

    if args.combine:
        run_dirs = [Path(p.strip()) for p in args.combine.split(",") if p.strip()]
        out_dir = _RESULTS_ROOT / f"combined_{_now_stamp()}"
        combine_runs(run_dirs, out_dir)
        return

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in ROSTER]
    if unknown:
        parser.error(f"unknown method(s): {', '.join(unknown)}; available: {', '.join(ROSTER)}")

    _load_env()
    out_dir = _RESULTS_ROOT / f"{args.model.replace('/', '-')}_{_now_stamp()}"
    print(f"model {args.model}  methods {methods}  judge {args.judge}\n-> {out_dir}", flush=True)
    run_head_to_head(methods, model=args.model, judge=args.judge, out_dir=out_dir)
    print(f"done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
