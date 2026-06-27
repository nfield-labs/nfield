"""Aggregate a result directory into a flat summary table (and optional plots).

Reads the per-record raw JSON arrays the runner wrote, flattens them to one row
per ``(method, fixture, seed)``, and emits ``summary.csv`` plus a console table.
Plotting is optional: matplotlib is imported only when a plot is requested, so
the aggregator runs with no extra dependency.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = ["Row", "collect_rows", "plot_va_vs_n", "write_summary_csv"]

# Coverage is the primary metric (how many fields the system surfaces vs leaves
# NULL); value_accuracy follows as the secondary correctness check.
_CSV_COLUMNS = (
    "budget",
    "method",
    "fixture",
    "seed",
    "n_fields",
    "coverage",
    "value_accuracy",
    "k",
    "k_min",
    "optimality_gap",
    "elapsed_seconds",
    "error_category",
    "error",
)

# The summary is a glanceable table; full error bodies (which can be multi-KB SDK
# reprs containing document text) live in the raw/ sidecars, not here.
_ERROR_MAX_CHARS: int = 160


@dataclass(frozen=True, slots=True)
class Row:
    """One flattened benchmark datapoint: a single ``(budget, method, fixture, seed)`` run.

    Value Accuracy is ``None`` for coverage-only fixtures (no gold key).
    ``budget`` is the budget-mode label (e.g. ``native``), empty for legacy runs.
    """

    budget: str
    method: str
    fixture: str
    seed: int
    n_fields: int
    value_accuracy: float | None
    coverage: float
    k: int
    k_min: int
    optimality_gap: float
    elapsed_seconds: float
    error_category: str | None
    error: str | None


def collect_rows(result_dir: Path) -> list[Row]:
    """Read every raw JSON array under ``result_dir`` into flat rows.

    Handles both layouts: per-budget subfolders (``result_dir/<budget>/raw/``,
    the runner's sweep output) and a single flat ``result_dir/raw/`` (legacy /
    single run). Each row's budget comes from the record itself.

    Args:
        result_dir: A result directory written by the runner.

    Returns:
        Rows sorted by ``(budget, method, fixture, n_fields, seed)``.

    Raises:
        FileNotFoundError: If no ``raw/`` outputs are found under ``result_dir``.
    """
    raw_files = sorted(result_dir.glob("*/raw/*.json"))
    if not raw_files and (result_dir / "raw").is_dir():
        raw_files = sorted((result_dir / "raw").glob("*.json"))
    if not raw_files:
        raise FileNotFoundError(f"no raw/ outputs under {result_dir}")

    rows: list[Row] = [
        _row(record)
        for sidecar in raw_files
        for record in json.loads(sidecar.read_text(encoding="utf-8"))
    ]
    rows.sort(key=lambda r: (r.budget, r.method, r.fixture, r.n_fields, r.seed))
    return rows


def write_summary_csv(rows: list[Row], path: Path) -> None:
    """Write ``rows`` to ``path`` as ``summary.csv`` with a stable column order."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            record = {key: asdict(row)[key] for key in _CSV_COLUMNS}
            record["error"] = _clamp_error(row.error)
            writer.writerow(record)


def _clamp_error(error: str | None) -> str | None:
    if error is None:
        return None
    flat = " ".join(error.split())
    return flat if len(flat) <= _ERROR_MAX_CHARS else flat[:_ERROR_MAX_CHARS] + "…"


def format_table(rows: list[Row]) -> str:
    """Render ``rows`` as a fixed-width console table.

    Coverage is the primary metric (fields filled vs left NULL); Value Accuracy
    follows as the secondary correctness check. Both are always shown.
    """
    header = (
        f"{'budget':<13}{'method':<17}{'fixture':<16}{'N':>6}{'filled':>8}{'null':>6}"
        f"{'cov':>8}{'VA':>8}{'K':>5}{'sec':>8}  why"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        filled = round(row.coverage * row.n_fields)
        va = f"{row.value_accuracy:.3f}" if row.value_accuracy is not None else "-"
        why = f"  {row.error_category}" if row.error_category else ""
        lines.append(
            f"{row.budget:<13}{row.method:<17}{row.fixture:<16}{row.n_fields:>6}{filled:>8}"
            f"{row.n_fields - filled:>6}{row.coverage:>8.3f}{va:>8}"
            f"{row.k:>5}{row.elapsed_seconds:>8.1f}{why}"
        )
    return "\n".join(lines)


def plot_va_vs_n(rows: list[Row], path: Path) -> Path | None:
    """Plot Value Accuracy vs N per method, one line per method.

    Args:
        rows: Flattened rows; rows without a Value Accuracy are skipped.
        path: Output image path (e.g. ``.../plots/va_vs_n.png``).

    Returns:
        The written path, or ``None`` if matplotlib is unavailable or no row
        carried a Value Accuracy.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    scored = [r for r in rows if r.value_accuracy is not None]
    if not scored:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    # One line per (method, budget) so the two budgets don't blur together.
    for method, budget in sorted({(r.method, r.budget) for r in scored}):
        series = sorted(
            (r.n_fields, r.value_accuracy)
            for r in scored
            if r.method == method and r.budget == budget and r.value_accuracy is not None
        )
        label = f"{method} [{budget}]" if budget else method
        axis.plot([n for n, _ in series], [va for _, va in series], marker="o", label=label)
    axis.set_xlabel("N (schema fields)")
    axis.set_ylabel("Value Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def _row(record: dict[str, object]) -> Row:
    return Row(
        budget=str(record.get("budget", "")),
        method=str(record.get("method", "")),
        fixture=str(record.get("fixture", "")),
        seed=int(_as_number(record.get("seed"), 0)),
        n_fields=int(_as_number(record.get("fields_total"), 0)),
        value_accuracy=_optional_float(record.get("value_accuracy")),
        coverage=float(_as_number(record.get("coverage"), 0.0)),
        k=int(_as_number(record.get("k"), 0)),
        k_min=int(_as_number(record.get("k_min"), 0)),
        optimality_gap=float(_as_number(record.get("optimality_gap"), 0.0)),
        elapsed_seconds=float(_as_number(record.get("elapsed_seconds"), 0.0)),
        error_category=_optional_str(record.get("error_category")),
        error=_optional_str(record.get("error")),
    )


def _as_number(value: object, default: float) -> float:
    return value if isinstance(value, int | float) else default


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m benchmark.report <result_dir>``."""
    parser = argparse.ArgumentParser(prog="benchmark.report", description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--plot", action="store_true", help="also write plots/va_vs_n.png")
    args = parser.parse_args(argv)

    rows = collect_rows(args.result_dir)
    summary_path = args.result_dir / "summary.csv"
    write_summary_csv(rows, summary_path)
    print(format_table(rows))
    print(f"\nsummary -> {summary_path}")
    if args.plot:
        plotted = plot_va_vs_n(rows, args.result_dir / "plots" / "va_vs_n.png")
        print(
            f"plot    -> {plotted}"
            if plotted
            else "plot    -> skipped (matplotlib absent / no VA)"
        )


if __name__ == "__main__":
    main()
