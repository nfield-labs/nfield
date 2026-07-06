"""Re-score a saved ExtractBench results folder under the benchmark's LLM tiers.

Extraction and scoring are separable: the sweep stores every raw extraction, so
the deterministic score can be upgraded to the judged score any time without
re-running a single extraction call. Deterministic misses on ``string_semantic``
fields and ``array_llm`` arrays are re-asked of an LLM judge; the judge can only
flip a miss to correct, so the judged score is always >= the deterministic one.

Usage::

    uv run python -m benchmark.rejudge_extractbench <results>/native [--model ...]

Writes ``scored_judged/<doc>.json`` beside each dataset's ``scored/`` and a
``summary_judged.csv`` per dataset plus one at the root.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any

from .runner_extractbench import _GOLD_SUFFIX, _SCHEMA_SUFFIX, discover_datasets
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
            _write_json(
                dataset_dir / "scored_judged" / f"{doc}.json",
                {
                    "document": doc,
                    "gold_fields": judged.n_fields,
                    "coverage": round(judged.coverage, 4),
                    "value_accuracy": round(judged.value_accuracy, 4),
                    "deterministic_value_accuracy": round(report.value_accuracy, 4),
                    "outcomes": {o.value: n for o, n in judged.outcomes.items()},
                    "mismatches": mismatches,
                },
            )
            row = {
                "dataset": name,
                "document": doc,
                "gold_fields": judged.n_fields,
                "coverage": round(judged.coverage, 4),
                "value_accuracy": round(judged.value_accuracy, 4),
                "deterministic": round(report.value_accuracy, 4),
            }
            rows.append(row)
            print(
                f"  {name}/{doc}: deterministic {row['deterministic']}"
                f" -> judged {row['value_accuracy']}",
                flush=True,
            )
        _write_rows(dataset_dir / "summary_judged.csv", rows)
        all_rows.extend(rows)
    _write_rows(results_dir / "summary_judged.csv", _dataset_aggregates(all_rows))


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
                "deterministic": round(
                    sum(r["deterministic"] * r["gold_fields"] for r in sub) / total, 4
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
    only = {n.strip() for n in args.datasets.split(",") if n.strip()} or None
    rejudge_results(args.results_dir, args.model, args.reasoning_model, only)


if __name__ == "__main__":
    main()
