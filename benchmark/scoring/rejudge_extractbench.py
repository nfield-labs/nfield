"""Re-score a saved ExtractBench results folder under the benchmark's LLM tiers.

Extraction and scoring are separable: the sweep stores every raw extraction, so
the deterministic score can be upgraded to the judged score any time without
re-running a single extraction call. Deterministic misses on ``string_semantic``
fields and ``array_llm`` arrays are re-asked of an LLM judge; the judge can only
flip a miss to correct, so the judged score is always >= the deterministic one.

Usage::

    uv run python -m benchmark.scoring.rejudge_extractbench <results>/native [--model ...]

Updates each dataset's ``scored/<doc>.json`` in place - ``value_accuracy_judged``
lands beside ``value_accuracy``, and the judged mismatch list replaces the
deterministic one (judge-cleared fields drop out). ``summary.csv`` rows gain a
``value_accuracy_judged`` column the same way.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any

from ..benchmarks.runner_extractbench import (
    _GOLD_SUFFIX,
    _SCHEMA_SUFFIX,
    _load_env,
    discover_datasets,
)
from .score_extractbench import llm_rejudge, score_extractbench

DEFAULT_JUDGE_MODEL = "groq/qwen/qwen3.6-27b"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _judge_complete(model: str, reasoning_model: bool) -> Any:
    from nfield.providers.groq._provider import GroqProvider

    judge = GroqProvider(model.split("/", 1)[1], reasoning_model=reasoning_model)

    async def complete(prompt: str) -> str:
        return await judge.complete([{"role": "user", "content": prompt}], max_tokens=2000)

    return complete


def rejudge_results(
    results_dir: Path, model: str, reasoning_model: bool, only: set[str] | None = None
) -> None:
    """Re-score every raw extraction under *results_dir* with the LLM judge."""
    datasets = discover_datasets()
    complete = _judge_complete(model, reasoning_model)
    all_rows: list[dict[str, Any]] = []
    for dataset_dir in sorted(p for p in results_dir.iterdir() if (p / "raw").is_dir()):
        name = dataset_dir.name
        if only and name not in only:
            continue
        base = datasets.get(name)
        if base is None:
            print(f"skipping {name}: no matching dataset")
            continue
        schema = json.loads((base / f"{base.name}{_SCHEMA_SUFFIX}").read_text(encoding="utf-8"))
        # A few dataset schemas wrap the JSON Schema in a descriptor object.
        schema = schema.get("schema_definition", schema)
        rows: list[dict[str, Any]] = []
        for raw_path in sorted((dataset_dir / "raw").glob("*.json")):
            doc = raw_path.stem
            gold_path = base / "pdf+gold" / f"{doc}{_GOLD_SUFFIX}"
            if not gold_path.exists():
                print(f"  {name}/{doc}: no gold file, skipped")
                continue
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            report, gold = score_extractbench(
                raw["data"], json.loads(gold_path.read_text(encoding="utf-8")), schema
            )
            judged = asyncio.run(llm_rejudge(report, gold, schema, complete))
            mismatches = [
                {
                    "path": fs.path,
                    "outcome": fs.outcome.value,
                    "gold": fs.gold,
                    "predicted": fs.predicted,
                }
                for fs in judged.fields
                if fs.outcome.value != "correct"
            ]
            scored_path = dataset_dir / "scored" / f"{doc}.json"
            scored = json.loads(scored_path.read_text(encoding="utf-8"))
            # The judged score lands right beside the deterministic one, and the
            # mismatch list shrinks to what the judge upheld.
            updated: dict[str, Any] = {}
            for key, value in scored.items():
                updated[key] = value
                if key == "value_accuracy":
                    updated["value_accuracy"] = round(report.value_accuracy, 4)
                    updated["value_accuracy_judged"] = round(judged.value_accuracy, 4)
            updated["outcomes"] = {o.value: n for o, n in judged.outcomes.items()}
            updated["mismatches"] = mismatches
            _write_json(scored_path, updated)
            row = {
                "dataset": name,
                "document": doc,
                "gold_fields": judged.n_fields,
                "coverage": round(judged.coverage, 4),
                "value_accuracy": round(report.value_accuracy, 4),
                "value_accuracy_judged": round(judged.value_accuracy, 4),
            }
            rows.append(row)
            print(
                f"  {name}/{doc}: deterministic {row['value_accuracy']}"
                f" -> judged {row['value_accuracy_judged']}",
                flush=True,
            )
        _merge_summary(dataset_dir / "summary.csv", "document", rows)
        all_rows.extend(rows)
    _merge_summary(results_dir / "summary.csv", "dataset", _dataset_aggregates(all_rows))


def _merge_summary(path: Path, key: str, rows: list[dict[str, Any]]) -> None:
    """Refresh the floor and add the judged column in an existing summary, in place."""
    if not path.exists() or not rows:
        return
    by_key = {str(r[key]): r for r in rows}
    with path.open(newline="", encoding="utf-8") as fh:
        existing = list(csv.DictReader(fh))
    for row in existing:
        update = by_key.get(row.get(key, ""))
        if update is None:
            row.setdefault("value_accuracy_judged", "")
            continue
        row["value_accuracy"] = update["value_accuracy"]
        row["value_accuracy_judged"] = update["value_accuracy_judged"]
    _write_rows(path, existing)


def _dataset_aggregates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Field-weighted judged aggregate per dataset, mirroring the sweep summary."""
    out: list[dict[str, Any]] = []
    for name in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == name]
        total = sum(r["gold_fields"] for r in sub)
        if not total:
            continue
        out.append(
            {
                "dataset": name,
                "documents": len(sub),
                "gold_fields": total,
                "coverage": round(sum(r["coverage"] * r["gold_fields"] for r in sub) / total, 4),
                "value_accuracy": round(
                    sum(r["value_accuracy"] * r["gold_fields"] for r in sub) / total, 4
                ),
                "value_accuracy_judged": round(
                    sum(r["value_accuracy_judged"] * r["gold_fields"] for r in sub) / total, 4
                ),
            }
        )
    return out


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="a sweep's native/ folder")
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--reasoning-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--datasets", default="", help="comma-separated names; empty = all")
    args = parser.parse_args(argv)
    _load_env()
    only = {n.strip() for n in args.datasets.split(",") if n.strip()} or None
    rejudge_results(args.results_dir, args.model, args.reasoning_model, only)


if __name__ == "__main__":
    main()
