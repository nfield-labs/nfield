#!/usr/bin/env python3
"""
Train and commit the ThresholdOracle from a completed benchmark summary CSV.

Run this **after** ``scripts/record_fixtures.py`` and a benchmark run to
produce a trained ``.pkl`` file that all contributors can use without
ever re-running the benchmark themselves.

Usage::

    # Default paths
    python scripts/train_oracle.py

    # Custom CSV
    python scripts/train_oracle.py --csv benchmark_results/summary.csv

    # Custom output
    python scripts/train_oracle.py --output \
        src/formatshield/oracle/oracle_data/threshold_oracle_v1.pkl

    # Validate CSV without training
    python scripts/train_oracle.py --dry-run

After training::

    git add src/formatshield/oracle/oracle_data/threshold_oracle_v1.pkl
    git commit -m 'feat: train threshold oracle from real Groq benchmark'

Contributors who pull this commit immediately get a working oracle —
no benchmark run required.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_DEFAULT_CSV = "benchmark_results/summary.csv"
_DEFAULT_OUTPUT = "src/formatshield/oracle/oracle_data/threshold_oracle_v1.pkl"
_MIN_ROWS = 10


def _validate_csv(csv_path: Path) -> list[dict[str, str]]:
    """Return parsed rows or exit with an informative error."""
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        print("Run the benchmark first:", file=sys.stderr)
        print(
            "  formatshield benchmark --tasks gsm,medical_ner,template_fill "
            "--backends groq --quick",
            file=sys.stderr,
        )
        sys.exit(1)

    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        print("ERROR: CSV is empty", file=sys.stderr)
        sys.exit(1)

    required_cols = {"accuracy_delta", "complexity_score"}
    missing = required_cols - set(rows[0].keys())
    if missing:
        print(f"ERROR: CSV missing required columns: {missing}", file=sys.stderr)
        print(f"Found columns: {list(rows[0].keys())}", file=sys.stderr)
        sys.exit(1)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ThresholdOracle from benchmark summary CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--csv",
        default=_DEFAULT_CSV,
        help=f"Path to benchmark summary CSV (default: {_DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help=f"Output .pkl path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CSV and report stats without training",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_path = Path(args.output)

    rows = _validate_csv(csv_path)

    # Summary stats
    n_positive = sum(1 for r in rows if float(r.get("accuracy_delta", 0)) > 0)
    print(f"CSV            : {csv_path} ({len(rows)} rows)")
    print(f"Positive labels: {n_positive} / {len(rows)} (TTF helped)")
    print(f"Negative labels: {len(rows) - n_positive} / {len(rows)} (TTF hurt or neutral)")

    if len(rows) < _MIN_ROWS:
        print(
            f"\nERROR: Need at least {_MIN_ROWS} rows to train a reliable oracle, got {len(rows)}",
            file=sys.stderr,
        )
        print("Tip: run without --quick or add more tasks", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("\nDry run — all checks passed, not training")
        return

    # Train
    print(f"\nOutput         : {output_path.absolute()}")
    print("\nTraining ThresholdOracle …")

    from formatshield.oracle.threshold_oracle import ThresholdOracle

    output_path.parent.mkdir(parents=True, exist_ok=True)
    oracle = ThresholdOracle.from_benchmark_data(csv_path, model_path=output_path, save=True)

    print(f"Oracle saved   : {output_path.absolute()}")
    try:
        print(f"Threshold      : {oracle._threshold:.4f}")
    except AttributeError:
        pass

    print()
    print("Next steps:")
    print("  Commit the oracle so contributors don't need to re-run the benchmark:")
    print(f"    git add {output_path}")
    print("    git commit -m 'feat: train threshold oracle from real Groq benchmark'")


if __name__ == "__main__":
    main()
