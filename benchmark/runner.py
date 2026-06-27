"""Sweep orchestrator - generate raw outputs, score them, and pin reproducibility.

Generation and scoring are deliberately decoupled: ``run`` writes per-record raw
outputs as readable indented JSON arrays (rewritten after each seed, so a crash
leaves a valid partial array) plus a ``MANIFEST.json`` that pins the model, date,
seeds, and library versions; ``score`` re-judges already-written raw outputs
against a gold key with no API call, so a rubric change never forces an expensive
regeneration.

Every method is swept under each budget profile (``benchmark.budget``); a run
writes one timestamped directory whose per-budget subfolders (``native/``,
``constrained/``) each hold ``raw/`` and ``scored/``, with one ``summary.csv``
(carrying a ``budget`` column) and one ``MANIFEST.json`` at the run root. Result
paths are self-describing, so a ``(budget, method, fixture, seed)`` cell is
locatable without a separate index.

This is a manual, budgeted tool. The real sweeps cost API calls; never wire the
full sweep into an automated job.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from . import datasets
from .adapters.instructor_adapter import InstructorAdapter
from .adapters.langchain_adapter import LangChainAdapter
from .adapters.native_json_adapter import NativeJsonAdapter
from .adapters.nfield_adapter import NfieldAdapter
from .adapters.raw_prompt_adapter import RawPromptAdapter
from .budget import BUDGET_MODES, BudgetMode, resolve_budget
from .score import score

if TYPE_CHECKING:
    from collections.abc import Callable

    from .adapters import Adapter, AdapterOutput
    from .datasets import LoadedDataset
    from .score import ScoreReport

__all__ = ["ADAPTERS", "RunArtifacts", "result_dir", "run_sweep", "score_existing"]

_DEFAULT_MODEL = "groq/llama-3.3-70b-versatile"
_RESULTS_ROOT = Path(__file__).resolve().parent / "results"

# The committed error field is a short diagnostic, not a raw SDK dump: provider
# SDK exceptions can be tens of KB and echo back document text / internal ids, so
# the stored error is whitespace-collapsed, org-id-redacted, and length-bounded.
# The full traceback stays on the console at run time.
_ERROR_MAX_CHARS: int = 300
_ORG_ID = re.compile(r"org_[A-Za-z0-9]{6,}")

# Method name -> zero-arg factory. nfield plus the Track-A (orchestration-layer)
# baselines, all on the same hosted model. Each is a thin Adapter wrapper.
ADAPTERS: dict[str, Callable[[], Adapter]] = {
    "nfield": NfieldAdapter,
    "raw_prompt": RawPromptAdapter,
    "native_json": NativeJsonAdapter,
    "instructor": InstructorAdapter,
    "langchain": LangChainAdapter,
}


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Paths written by one ``(method, fixture)`` sweep.

    Args:
        raw_path: The per-record raw JSON array.
        scored_path: The aggregated score JSON, or ``None`` when the fixture has
            no gold key (coverage-only).
        manifest_path: The reproducibility manifest for the result directory.
    """

    raw_path: Path
    scored_path: Path | None
    manifest_path: Path


def result_dir(model: str, stamp: str, *, root: Path = _RESULTS_ROOT) -> Path:
    """Return the result directory for a model+stamp, e.g. ``groq-llama_2026-06-09_14-30-05``.

    The stamp carries date AND time so two runs on the same day land in distinct
    folders instead of overwriting; the format sorts lexically, so the newest run
    is the last directory by name.
    """
    return root / f"{model.replace('/', '-')}_{stamp}"


def _now_stamp() -> str:
    """A filesystem-safe, lexically sortable UTC run stamp: ``YYYY-MM-DD_HH-MM-SS``."""
    return dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def _latest_stamp(model: str, *, root: Path = _RESULTS_ROOT) -> str | None:
    """Return the newest existing run stamp for ``model``, or ``None`` if there is none."""
    prefix = f"{model.replace('/', '-')}_"
    stamps = sorted(p.name[len(prefix) :] for p in root.glob(f"{prefix}*") if p.is_dir())
    return stamps[-1] if stamps else None


def run_sweep(
    adapter: Adapter,
    dataset: LoadedDataset,
    *,
    model: str,
    seeds: int,
    out_dir: Path,
    context_window: int,
    max_output_tokens: int,
    budget: str = "",
    manifest_dir: Path | None = None,
) -> RunArtifacts:
    """Run one method over one fixture for ``seeds`` repeats and persist results.

    Args:
        adapter: The method under test.
        dataset: The loaded fixture (schema + document + optional gold).
        model: Provider-qualified model id used for the run and the result path.
        seeds: Number of repeats (temp=0 is still stochastic, so variance is
            measured by repetition).
        out_dir: Where ``raw/`` and ``scored/`` are written (the per-budget dir);
            created if absent.
        context_window: Shared input-window budget passed to the method.
        max_output_tokens: Shared output budget passed to the method.
        budget: Budget-mode label recorded on every row (e.g. ``"native"``).
        manifest_dir: Where ``MANIFEST.json`` is written; defaults to ``out_dir``.
            The runner points it at the run root so one manifest spans all budgets.

    Returns:
        The :class:`RunArtifacts` paths written.
    """
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{adapter.name}_{dataset.name}.json"

    reports: list[ScoreReport] = []
    records: list[dict[str, Any]] = []
    for seed in range(seeds):
        output = adapter.run(
            dataset.document,
            dataset.schema,
            model=model,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            instructions=dataset.instructions,
        )
        report = (
            score(output.data, dataset.gold, dataset.schema, call_failed=output.call_failed)
            if dataset.gold
            else None
        )
        if report is not None:
            reports.append(report)
        records.append(_record(adapter.name, dataset.name, seed, output, report, budget=budget))
        # Rewrite the whole indented array after each seed: readable JSON, and
        # still durable (a crash leaves a valid array of the seeds done so far).
        raw_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    scored_path = _write_scored(out_dir, adapter.name, dataset, reports) if reports else None
    manifest_path = _write_manifest(
        manifest_dir or out_dir, model, adapter.name, dataset, seeds, budget=budget
    )
    return RunArtifacts(raw_path=raw_path, scored_path=scored_path, manifest_path=manifest_path)


def score_existing(raw_path: Path, dataset: LoadedDataset) -> ScoreReport:
    """Re-score the last record of an existing raw JSON array against the gold key.

    Args:
        raw_path: A raw JSON array previously written by :func:`run_sweep`.
        dataset: The fixture supplying the gold key and schema.

    Returns:
        A fresh :class:`ScoreReport` for the most recent record.

    Raises:
        ValueError: If the fixture has no gold key, or the sidecar is empty.
    """
    if dataset.gold is None:
        raise ValueError(f"dataset {dataset.name!r} has no gold key to score against")
    records = json.loads(raw_path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"no records in {raw_path}")
    last = records[-1]
    return score(
        last.get("data", {}),
        dataset.gold,
        dataset.schema,
        call_failed=int(last.get("call_failed", 0)),
    )


def _record(
    method: str,
    fixture: str,
    seed: int,
    output: AdapterOutput,
    report: ScoreReport | None,
    *,
    budget: str = "",
) -> dict[str, Any]:
    # Coverage is gold-based (fraction of the answer key filled, capped at 1.0)
    # when a gold key exists; only for coverage-only fixtures does it fall back to
    # the adapter's raw leaf count, which can exceed N if the model adds keys.
    coverage = (
        report.coverage
        if report is not None
        else _safe_ratio(output.fields_extracted, output.fields_total)
    )
    record: dict[str, Any] = {
        "method": method,
        "fixture": fixture,
        "budget": budget,
        "seed": seed,
        "fields_total": output.fields_total,
        "fields_extracted": output.fields_extracted,
        "coverage": coverage,
        "k": output.k,
        "k_min": output.k_min,
        "optimality_gap": _safe_ratio(output.k - output.k_min, output.k_min),
        "call_failed": output.call_failed,
        "elapsed_seconds": output.elapsed_seconds,
        "error": _sanitize_error(output.error),
        "error_category": output.error_category,
        "data": output.data,
    }
    if report is not None:
        record["value_accuracy"] = report.value_accuracy
        record["by_type"] = {ft.value: ts.accuracy for ft, ts in report.by_type.items()}
        record["outcomes"] = {o.value: c for o, c in report.outcomes.items()}
    return record


def _sanitize_error(error: str | None) -> str | None:
    if error is None:
        return None
    flat = _ORG_ID.sub("org_<redacted>", " ".join(error.split()))
    return flat if len(flat) <= _ERROR_MAX_CHARS else flat[:_ERROR_MAX_CHARS] + "…"


def _write_scored(
    out_dir: Path,
    method: str,
    dataset: LoadedDataset,
    reports: list[ScoreReport],
) -> Path:
    scored_dir = out_dir / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    path = scored_dir / f"{method}_{dataset.name}.json"
    n_fields = reports[0].n_fields
    coverage_mean = _mean([r.coverage for r in reports])
    accuracies = [r.value_accuracy for r in reports]
    # Coverage is the primary metric (how many fields the system surfaces vs
    # leaves NULL - what the decomposition/retrieval architecture drives); Value
    # Accuracy is reported alongside as the secondary check that the surfaced
    # values are also correct.
    payload = {
        "method": method,
        "fixture": dataset.name,
        "n_fields": n_fields,
        "runs": len(reports),
        "coverage_mean": coverage_mean,
        "fields_covered_mean": round(coverage_mean * n_fields, 1),
        "value_accuracy_mean": _mean(accuracies),
        "value_accuracy_std": _std(accuracies),
        "precision_mean": _mean([r.precision for r in reports]),
        "reliability_mean": _mean([r.reliability for r in reports]),
        "json_pass_all": all(r.json_pass for r in reports),
        "by_type_mean": _by_type_mean(reports),
        "call_failed_mean": _mean([float(r.call_failed) for r in reports]),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_manifest(
    out_dir: Path,
    model: str,
    method: str,
    dataset: LoadedDataset,
    seeds: int,
    *,
    budget: str = "",
) -> Path:
    from nfield import __version__ as fs_version

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "MANIFEST.json"
    manifest = _load_json(path)
    manifest.setdefault("model", model)
    manifest.setdefault("date", dt.datetime.now(tz=dt.timezone.utc).date().isoformat())
    manifest.setdefault("temperature", 0)
    manifest["library_version"] = fs_version
    manifest["python_version"] = sys.version.split()[0]
    runs = manifest.setdefault("runs", [])
    runs.append(
        {
            "method": method,
            "fixture": dataset.name,
            "budget": budget,
            "n_fields": _gold_or_schema_count(dataset),
            "seeds": seeds,
            "schema_sha256": _hash(json.dumps(dataset.schema, sort_keys=True)),
            "document_sha256": _hash(dataset.document),
            "recorded_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds"),
        }
    )
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _by_type_mean(reports: list[ScoreReport]) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for report in reports:
        for ft, stat in report.by_type.items():
            totals.setdefault(ft.value, []).append(stat.accuracy)
    return {name: _mean(values) for name, values in totals.items()}


def _gold_or_schema_count(dataset: LoadedDataset) -> int:
    return len(dataset.gold) if dataset.gold is not None else 0


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return float((sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _build_adapter(method: str) -> Adapter:
    try:
        factory = ADAPTERS[method]
    except KeyError:
        raise SystemExit(
            f"unknown method {method!r}; available: {', '.join(sorted(ADAPTERS))}"
        ) from None
    return factory()


def _validated_budgets(spec: str) -> list[BudgetMode]:
    # De-duplicate while preserving order: a repeated budget would otherwise run
    # the whole matrix again into the same paths (wasted API calls).
    budgets = list(dict.fromkeys(b.strip() for b in spec.split(",") if b.strip()))
    unknown = [b for b in budgets if b not in BUDGET_MODES]
    if unknown or not budgets:
        raise SystemExit(
            f"unknown budget {unknown or spec!r}; choose from {', '.join(BUDGET_MODES)}"
        )
    return [cast("BudgetMode", b) for b in budgets]


def _cmd_run(args: argparse.Namespace) -> None:
    dataset = datasets.get(args.fixture).load()
    adapter = _build_adapter(args.method)
    budget = _validated_budgets(args.budget)[0]
    limits = resolve_budget(budget, args.model)
    run_root = result_dir(args.model, args.date or _now_stamp())
    artifacts = run_sweep(
        adapter,
        dataset,
        model=args.model,
        seeds=args.seeds,
        out_dir=run_root / budget,
        context_window=limits.context_window,
        max_output_tokens=limits.max_output_tokens,
        budget=budget,
        manifest_dir=run_root,
    )
    print(f"raw      -> {artifacts.raw_path}")
    if artifacts.scored_path is not None:
        print(f"scored   -> {artifacts.scored_path}")
        print(json.dumps(_load_json(artifacts.scored_path), indent=2))
    else:
        print(f"(no gold key for {dataset.name}; coverage-only, not scored)")
    print(f"manifest -> {artifacts.manifest_path}")


def _cmd_sweep(args: argparse.Namespace) -> None:
    from . import report

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    fixtures = [f.strip() for f in args.fixtures.split(",") if f.strip()]
    budgets = _validated_budgets(args.budgets)
    run_root = result_dir(args.model, args.date or _now_stamp())

    for budget in budgets:
        limits = resolve_budget(budget, args.model)
        for fixture in fixtures:
            dataset = datasets.get(fixture).load()
            for method in methods:
                adapter = _build_adapter(method)
                print(
                    f"  [{budget}] running {method} x {fixture} (seeds={args.seeds}) ...",
                    flush=True,
                )
                run_sweep(
                    adapter,
                    dataset,
                    model=args.model,
                    seeds=args.seeds,
                    out_dir=run_root / budget,
                    context_window=limits.context_window,
                    max_output_tokens=limits.max_output_tokens,
                    budget=budget,
                    manifest_dir=run_root,
                )

    rows = report.collect_rows(run_root)
    report.write_summary_csv(rows, run_root / "summary.csv")
    report.plot_va_vs_n(rows, run_root / "plots" / "va_vs_n.png")
    print("\n" + report.format_table(rows))
    print(f"\nresults -> {run_root}")


def _cmd_score(args: argparse.Namespace) -> None:
    dataset = datasets.get(args.fixture).load()
    budget = _validated_budgets(args.budget)[0]
    stamp = args.date or _latest_stamp(args.model)
    if stamp is None:
        raise SystemExit(f"no existing run for {args.model}; run it first")
    raw_path = (
        result_dir(args.model, stamp) / budget / "raw" / f"{args.method}_{dataset.name}.json"
    )
    if not raw_path.exists():
        raise SystemExit(f"no raw output at {raw_path}; run it first")
    report = score_existing(raw_path, dataset)
    covered = round(report.coverage * report.n_fields)
    print(
        f"coverage       = {report.coverage:.3f}  ({covered}/{report.n_fields} filled, "
        f"{report.n_fields - covered} NULL)"
    )
    print(f"value_accuracy = {report.value_accuracy:.3f}  (secondary)")
    for ft, stat in report.by_type.items():
        print(f"  {ft.value:<14} {stat.correct}/{stat.total}  ({stat.accuracy:.3f})")
    print("outcomes:", {o.value: c for o, c in report.outcomes.items()})


def _load_env() -> None:
    # Manual convenience: a live sweep needs GROQ_API_KEY in the environment.
    # Mirror the repo's other live tooling and read it from a local .env.
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
    """Entry point for ``python -m benchmark.runner``."""
    _load_env()
    parser = argparse.ArgumentParser(prog="benchmark.runner", description=__doc__)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument(
        "--date",
        default=None,
        help="override the result-dir stamp (e.g. 2026-06-09_14-30-05); "
        "run/sweep default to now, score defaults to the latest existing run",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a method over a fixture and score it (costs API calls)")
    run.add_argument("--method", required=True, choices=sorted(ADAPTERS))
    run.add_argument("--fixture", required=True, choices=datasets.available())
    run.add_argument("--seeds", type=int, default=1)
    run.add_argument("--budget", default=BUDGET_MODES[0], choices=BUDGET_MODES)
    run.set_defaults(func=_cmd_run)

    sweep = sub.add_parser(
        "sweep", help="run a method x fixture matrix and aggregate (costs calls)"
    )
    sweep.add_argument(
        "--methods", default=",".join(ADAPTERS), help="comma-separated method names"
    )
    sweep.add_argument(
        "--fixtures", default=",".join(datasets.available()), help="comma-separated fixture names"
    )
    sweep.add_argument("--seeds", type=int, default=1)
    sweep.add_argument(
        "--budgets",
        default=",".join(BUDGET_MODES),
        help=f"comma-separated budget modes ({', '.join(BUDGET_MODES)})",
    )
    sweep.set_defaults(func=_cmd_sweep)

    rescore = sub.add_parser("score", help="re-score an existing raw output (no API)")
    rescore.add_argument("--method", required=True)
    rescore.add_argument("--fixture", required=True, choices=datasets.available())
    rescore.add_argument("--budget", default=BUDGET_MODES[0], choices=BUDGET_MODES)
    rescore.set_defaults(func=_cmd_score)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
