#!/usr/bin/env python3
"""
FormatShield - Groq-only end-to-end oracle training pipeline.

Runs the full pipeline in one shot:

    Step 1  Record real Groq responses -> tests/fixtures/groq_responses.jsonl
    Step 2  Benchmark with LLM judge   -> benchmark_results/summary.csv
    Step 3  Train oracle               -> oracle_data/threshold_oracle_v1.pkl
    Step 4  Validation report          -> benchmark_results/validation_report.json
    Step 5  Print git commit commands

Usage::

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --tasks gsm_symbolic,medical_ner,template_fill
    python scripts/run_pipeline.py --generator llama-3.1-8b-instant --judge llama-3.3-70b-versatile

The script is idempotent: running it twice appends new fixtures and
regenerates the oracle from the combined CSV.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Allow running from repo root without package install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import formatshield  # noqa: F401 - auto-loads .env

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "groq_responses.jsonl"
BENCH_DIR = ROOT / "benchmark_results"
ORACLE_PATH = ROOT / "src" / "formatshield" / "oracle" / "oracle_data" / "threshold_oracle_v1.pkl"
JUDGE_CACHE = ROOT / "benchmark_results" / "judge_cache"
REPORT_PATH = BENCH_DIR / "validation_report.json"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TASKS = "gsm_symbolic,medical_ner,template_fill,classification,financial,legal_extract"
DEFAULT_GENERATOR = "llama-3.1-8b-instant"   # cheap + fast
DEFAULT_JUDGE = "llama-3.3-70b-versatile"    # stronger for evaluation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_section(title: str) -> None:
    line = "-" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(line)


def _count_fixture_lines() -> int:
    if not FIXTURE_PATH.exists():
        return 0
    return sum(1 for _ in FIXTURE_PATH.open(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def run_pipeline(
    tasks: list[str],
    generator_model: str,
    judge_model: str,
    quick: bool,
    api_key: str,
) -> dict:
    from formatshield.backends.groq_backend import GroqBackend
    from formatshield.backends.replay_backend import RecordingBackend
    from formatshield.benchmark.harness import BenchmarkHarness
    from formatshield.benchmark.judge import LLMJudge

    report: dict = {
        "generator_model": generator_model,
        "judge_model": judge_model,
        "tasks": tasks,
        "quick": quick,
        "steps": {},
    }

    # ------------------------------------------------------------------
    # Step 1: Record fixtures
    # ------------------------------------------------------------------
    _print_section("Step 1 - Recording Groq fixtures")

    lines_before = _count_fixture_lines()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    generator_backend = GroqBackend(api_key=api_key, model=generator_model)
    recorder = RecordingBackend(generator_backend, fixture_path=FIXTURE_PATH)

    # ------------------------------------------------------------------
    # Step 2: Benchmark with judge
    # ------------------------------------------------------------------
    _print_section("Step 2 - Running benchmark with LLM judge")

    judge_backend = GroqBackend(api_key=api_key, model=judge_model)
    JUDGE_CACHE.mkdir(parents=True, exist_ok=True)
    judge = LLMJudge(backend=judge_backend, cache_dir=str(JUDGE_CACHE))

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    harness = BenchmarkHarness(output_dir=BENCH_DIR, judge=judge)

    print(f"Generator : groq/{generator_model}")
    print(f"Judge     : groq/{judge_model}")
    print(f"Tasks     : {', '.join(tasks)}")
    print(f"Quick     : {quick}")
    print(f"Fixture   : {FIXTURE_PATH}")

    results = await harness.run(
        tasks=tasks,
        backends=["groq"],
        models={"groq": f"groq/{generator_model}"},
        quick=quick,
        backend_objects={"groq": recorder},
    )

    lines_after = _count_fixture_lines()
    new_records = lines_after - lines_before

    print(f"\nBenchmark complete - {len(results)} result(s)")
    print(f"New fixture records : {new_records} (total: {lines_after})")

    report["steps"]["benchmark"] = {
        "results_count": len(results),
        "fixture_records_added": new_records,
        "fixture_total": lines_after,
    }

    # ------------------------------------------------------------------
    # Step 3: Train oracle - write per-problem CSV then train
    # ------------------------------------------------------------------
    _print_section("Step 3 - Training oracle")

    # summary.csv is aggregated (one row per task) - not enough rows for
    # the oracle to train on.  Write a per-problem CSV from the results list.
    import csv as csv_mod

    per_problem_csv = BENCH_DIR / "per_problem.csv"
    if results:
        fieldnames = list(results[0].to_dict().keys())
        with per_problem_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv_mod.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())
        print(f"Per-problem CSV    : {per_problem_csv} ({len(results)} rows)")

    n_pos = sum(1 for r in results if r.accuracy_delta > 0)
    n_neg = sum(1 for r in results if r.accuracy_delta <= 0)
    print(f"Label distribution: TTF helped={n_pos} / did not={n_neg}")

    if not per_problem_csv.exists() or len(results) < 10:
        print(f"WARNING: Only {len(results)} rows - need >= 10 to train oracle. Skipping.")
        report["steps"]["oracle"] = {
            "status": "skipped",
            "reason": f"only {len(results)} rows (need 10)",
        }
    elif n_pos == 0 or n_neg == 0:
        # Single-class data - logistic regression cannot train
        # This happens when the model gets everything right (or wrong) in quick mode.
        print("WARNING: All accuracy_delta values are the same class.")
        print("  The model may be getting every quick-mode problem correct.")
        print("  Re-run with --no-quick or add harder tasks (zebralogic) for variance.")
        report["steps"]["oracle"] = {
            "status": "skipped",
            "reason": "single-class labels - run with --no-quick for harder problems",
            "positive_labels": n_pos,
            "negative_labels": n_neg,
        }
    else:
        from formatshield.oracle.threshold_oracle import ThresholdOracle

        ORACLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        ThresholdOracle.from_benchmark_data(
            per_problem_csv, model_path=ORACLE_PATH, save=True
        )
        print(f"Oracle saved       : {ORACLE_PATH}")
        print(f"Labels (TTF helped / did not): {n_pos} / {n_neg}")
        report["steps"]["oracle"] = {
            "status": "trained",
            "pkl_path": str(ORACLE_PATH),
            "rows": len(results),
            "positive_labels": n_pos,
            "negative_labels": n_neg,
        }

    # ------------------------------------------------------------------
    # Step 4: Validation report
    # ------------------------------------------------------------------
    _print_section("Step 4 - Generating validation report")

    task_metrics: dict[str, dict] = {}
    for r in results:
        key = r.task
        if key not in task_metrics:
            task_metrics[key] = {
                "direct_accuracy": [],
                "ttf_accuracy": [],
                "accuracy_delta": [],
                "overhead_pct": [],
            }
        task_metrics[key]["direct_accuracy"].append(r.direct_accuracy)
        task_metrics[key]["ttf_accuracy"].append(r.ttf_accuracy)
        task_metrics[key]["accuracy_delta"].append(r.accuracy_delta)
        task_metrics[key]["overhead_pct"].append(r.overhead_pct)

    aggregated: dict[str, dict] = {}
    for task_name, metrics in task_metrics.items():
        n = len(metrics["direct_accuracy"])
        aggregated[task_name] = {
            "n_problems": n,
            "direct_accuracy": round(sum(metrics["direct_accuracy"]) / n, 4),
            "ttf_accuracy": round(sum(metrics["ttf_accuracy"]) / n, 4),
            "accuracy_delta": round(sum(metrics["accuracy_delta"]) / n, 4),
            "overhead_pct": round(sum(metrics["overhead_pct"]) / n, 2),
        }

    overall_delta = sum(m["accuracy_delta"] for m in aggregated.values()) / max(len(aggregated), 1)
    report["task_metrics"] = aggregated
    report["overall_accuracy_delta"] = round(overall_delta, 4)

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Validation report  : {REPORT_PATH}")

    # Print table
    print()
    print(f"{'Task':<22} {'N':>4} {'Direct':>8} {'TTF':>8} {'Delta':>8} {'Overhead':>10}")
    print("-" * 64)
    for task_name, m in aggregated.items():
        delta_str = f"{m['accuracy_delta']:+.4f}"
        print(
            f"{task_name:<22} {m['n_problems']:>4} "
            f"{m['direct_accuracy']:>8.4f} {m['ttf_accuracy']:>8.4f} "
            f"{delta_str:>8} {m['overhead_pct']:>9.1f}%"
        )
    print("-" * 64)
    print(f"{'OVERALL':<22} {'':>4} {'':>8} {'':>8} {overall_delta:>+8.4f}")

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FormatShield Groq-only oracle training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="Comma-separated task names")
    parser.add_argument("--generator", default=DEFAULT_GENERATOR, help="Groq generator model")
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help="Groq judge model")
    parser.add_argument(
        "--quick", action="store_true", default=True, help="Quick mode (default: on)"
    )
    parser.add_argument("--no-quick", dest="quick", action="store_false", help="Full problem set")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print("FormatShield - Groq-only oracle training pipeline")
    print(f"Tasks     : {', '.join(task_list)}")
    print(f"Generator : groq/{args.generator}")
    print(f"Judge     : groq/{args.judge}")
    print(f"Quick     : {args.quick}")

    report = asyncio.run(
        run_pipeline(
            tasks=task_list,
            generator_model=args.generator,
            judge_model=args.judge,
            quick=args.quick,
            api_key=api_key,
        )
    )

    # ------------------------------------------------------------------
    # Step 5: Git instructions
    # ------------------------------------------------------------------
    _print_section("Step 5 - Git commit commands")

    files_to_add = [
        "tests/fixtures/groq_responses.jsonl",
        "src/formatshield/oracle/oracle_data/threshold_oracle_v1.pkl",
        "benchmark_results/validation_report.json",
    ]

    print()
    for f in files_to_add:
        if Path(ROOT / f).exists():
            print(f"  git add {f}")
    print()
    print('  git commit -m "feat: real Groq benchmark data, trained oracle, validation report"')

    oracle_status = report.get("steps", {}).get("oracle", {}).get("status", "unknown")
    print(f"\nPipeline complete. Oracle status: {oracle_status}")

    if oracle_status == "trained":
        rows = report["steps"]["oracle"].get("rows", 0)
        delta = report.get("overall_accuracy_delta", 0)
        print(f"Oracle trained on {rows} rows. Overall accuracy delta: {delta:+.4f}")
    elif oracle_status == "skipped":
        reason = report["steps"]["oracle"].get("reason", "")
        print(f"Oracle skipped: {reason}")
        print("Run with more tasks or without --quick to get >=10 rows.")


if __name__ == "__main__":
    main()
