#!/usr/bin/env python3
"""
Record golden fixture responses from a real Groq backend.

Run this script **once** to populate ``tests/fixtures/groq_responses.jsonl``.
After that, unit tests use :class:`ReplayBackend` to replay the stored
responses — no API key required on subsequent runs.

Usage::

    # Minimal (uses default tasks, smaller model, quick mode)
    GROQ_API_KEY=xxx python scripts/record_fixtures.py --quick

    # Specific tasks
    python scripts/record_fixtures.py --tasks gsm,medical_ner,template_fill

    # Full run (more data for oracle training)
    python scripts/record_fixtures.py --tasks gsm_symbolic,math500,legal_extract,sql_extraction

    # Custom output path
    python scripts/record_fixtures.py --output path/to/fixture.jsonl

Fixture format (one JSON object per line)::

    {"key": "<sha256>", "prompt": "...", "schema": {...}, "constraints": "json", "response": "..."}
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import formatshield  # noqa: F401 — triggers .env auto-load


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record Groq API responses to a JSONL fixture file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tasks",
        default="gsm,medical_ner,template_fill",
        help="Comma-separated task names (default: gsm,medical_ner,template_fill)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the reduced problem set — faster, fewer tokens consumed",
    )
    parser.add_argument(
        "--output",
        default="tests/fixtures/groq_responses.jsonl",
        help="Output fixture path (default: tests/fixtures/groq_responses.jsonl)",
    )
    parser.add_argument(
        "--model",
        default="llama-3.1-8b-instant",
        help="Groq model name (default: llama-3.1-8b-instant — cheapest/fastest)",
    )
    args = parser.parse_args()

    import os

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print(
            "ERROR: GROQ_API_KEY is not set.\n"
            "Add it to your .env file or run:\n"
            "  export GROQ_API_KEY=gsk-...\n"
            "  python scripts/record_fixtures.py",
            file=sys.stderr,
        )
        sys.exit(1)

    from formatshield.backends.groq_backend import GroqBackend
    from formatshield.backends.replay_backend import RecordingBackend
    from formatshield.benchmark.harness import BenchmarkHarness

    fixture_path = Path(args.output)
    fixture_path.parent.mkdir(parents=True, exist_ok=True)

    lines_before = _count_lines(fixture_path)

    real_backend = GroqBackend(api_key=api_key, model=args.model)
    recorder = RecordingBackend(real_backend, fixture_path=fixture_path)

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    harness = BenchmarkHarness(output_dir=Path("benchmark_results"))

    async def _run() -> None:
        print("FormatShield — fixture recorder")
        print(f"  Tasks   : {', '.join(task_list)}")
        print(f"  Model   : groq/{args.model}")
        print(f"  Quick   : {args.quick}")
        print(f"  Output  : {fixture_path.absolute()}")
        print()

        results = await harness.run(
            tasks=task_list,
            backends=["groq"],
            models={"groq": f"groq/{args.model}"},
            quick=args.quick,
            backend_objects={"groq": recorder},
        )

        lines_after = _count_lines(fixture_path)
        new_records = lines_after - lines_before

        print(f"\nComplete — {len(results)} benchmark result(s) recorded.")
        print(f"New fixture entries : {new_records}")
        print(f"Total fixture entries: {lines_after}")
        print(f"Fixture file        : {fixture_path.absolute()}")
        print()
        print("Next steps:")
        print("  1. Run unit tests (no API key needed):")
        print("       pytest tests/unit/ -v")
        print("  2. Train the oracle:")
        print("       python scripts/train_oracle.py")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
