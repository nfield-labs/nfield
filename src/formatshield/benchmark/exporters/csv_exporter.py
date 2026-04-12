"""
CSVExporter — writes FormatShield benchmark results to CSV files and LaTeX tables.

All I/O is handled via the Python standard-library ``csv`` module; no external
dependencies (pandas, numpy) are required for this module.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from formatshield.scorer.features import BenchmarkResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSV column ordering
# ---------------------------------------------------------------------------

_RESULT_FIELDS: list[str] = [
    "task",
    "backend",
    "model",
    "direct_accuracy",
    "ttf_accuracy",
    "accuracy_delta",
    "direct_latency_ms",
    "ttf_latency_ms",
    "overhead_pct",
    "complexity_score",
    "failure_modes_detected",
]

_SUMMARY_FIELDS: list[str] = [
    "backend",
    "task",
    "n_problems",
    "mean_direct_accuracy",
    "mean_ttf_accuracy",
    "mean_accuracy_delta",
    "mean_direct_latency_ms",
    "mean_ttf_latency_ms",
    "mean_overhead_pct",
    "mean_complexity_score",
]


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean of *values*, or 0.0 for empty sequences."""
    return sum(values) / len(values) if values else 0.0


class CSVExporter:
    """
    Export :class:`~formatshield.scorer.features.BenchmarkResult` objects
    to CSV files and generate LaTeX table code for academic papers.

    All methods are stateless and can be called on a single instance
    multiple times with different result sets.

    Example::

        exporter = CSVExporter()
        raw_path = exporter.export(results, Path("out/raw.csv"))
        summary_path = exporter.export_summary(results, Path("out/summary.csv"))
        latex = exporter.generate_latex_table(results)
        print(latex)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(
        self,
        results: list[BenchmarkResult],
        output_path: Path,
    ) -> Path:
        """
        Write all benchmark results to a CSV file, one row per result.

        Parameters
        ----------
        results:
            List of :class:`BenchmarkResult` instances to serialise.
        output_path:
            Destination file path.  Parent directories must already exist.

        Returns
        -------
        Path
            The path to the written CSV file (same as *output_path*).

        Raises
        ------
        OSError
            If *output_path* cannot be opened for writing.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_RESULT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for result in results:
                writer.writerow(self._result_to_row(result))

        logger.info("CSVExporter.export: wrote %d rows to %s", len(results), output_path)
        return output_path

    def export_summary(
        self,
        results: list[BenchmarkResult],
        output_path: Path,
    ) -> Path:
        """
        Aggregate results by (backend, task) and write a summary CSV.

        Each row in the summary represents one backend × task combination.
        Numeric columns are the arithmetic mean across all problems in that
        group.

        Parameters
        ----------
        results:
            List of :class:`BenchmarkResult` instances to aggregate.
        output_path:
            Destination file path.  Parent directories must already exist.

        Returns
        -------
        Path
            The path to the written summary CSV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        groups: dict[tuple[str, str], list[BenchmarkResult]] = defaultdict(list)
        for r in results:
            groups[(r.backend, r.task)].append(r)

        rows: list[dict[str, Any]] = []
        for (backend, task), group in sorted(groups.items()):
            rows.append(
                {
                    "backend": backend,
                    "task": task,
                    "n_problems": len(group),
                    "mean_direct_accuracy": round(_mean([r.direct_accuracy for r in group]), 4),
                    "mean_ttf_accuracy": round(_mean([r.ttf_accuracy for r in group]), 4),
                    "mean_accuracy_delta": round(_mean([r.accuracy_delta for r in group]), 4),
                    "mean_direct_latency_ms": round(_mean([r.direct_latency_ms for r in group]), 2),
                    "mean_ttf_latency_ms": round(_mean([r.ttf_latency_ms for r in group]), 2),
                    "mean_overhead_pct": round(_mean([r.overhead_pct for r in group]), 2),
                    "mean_complexity_score": round(_mean([r.complexity_score for r in group]), 4),
                }
            )

        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        logger.info(
            "CSVExporter.export_summary: wrote %d group rows to %s",
            len(rows),
            output_path,
        )
        return output_path

    def generate_latex_table(self, results: list[BenchmarkResult]) -> str:
        """
        Generate a LaTeX ``tabular`` environment comparing TTF vs direct
        accuracy across all backend × task combinations.

        The table has one column per task and one row per backend.  Each cell
        shows ``direct / ttf (Δ)`` accuracy values formatted to two decimal
        places.

        Parameters
        ----------
        results:
            List of :class:`BenchmarkResult` instances to tabulate.

        Returns
        -------
        str
            A complete LaTeX ``table`` environment (including caption and
            label) ready for inclusion in a ``.tex`` file.
        """
        # Collect unique backends and tasks in sorted order
        backends: list[str] = sorted({r.backend for r in results})
        tasks: list[str] = sorted({r.task for r in results})

        # Build lookup: (backend, task) → aggregated metrics
        cell_data: dict[tuple[str, str], dict[str, float]] = {}
        groups: dict[tuple[str, str], list[BenchmarkResult]] = defaultdict(list)
        for r in results:
            groups[(r.backend, r.task)].append(r)

        for (backend, task), group in groups.items():
            cell_data[(backend, task)] = {
                "direct": _mean([r.direct_accuracy for r in group]),
                "ttf": _mean([r.ttf_accuracy for r in group]),
                "delta": _mean([r.accuracy_delta for r in group]),
            }

        # Build column spec: backend col + one col per task
        col_spec = "l" + "c" * len(tasks)

        # Build header row
        task_headers = " & ".join(r"\textbf{" + t.replace("_", r"\_") + "}" for t in tasks)
        header_row = r"\textbf{Backend} & " + task_headers + r" \\"

        # Build data rows
        data_rows: list[str] = []
        for backend in backends:
            cells: list[str] = [backend.replace("_", r"\_")]
            for task in tasks:
                key = (backend, task)
                if key in cell_data:
                    d = cell_data[key]["direct"]
                    t = cell_data[key]["ttf"]
                    delta = cell_data[key]["delta"]
                    sign = "+" if delta >= 0 else ""
                    cells.append(rf"{d:.2f} / {t:.2f} ({sign}{delta:.2f})")
                else:
                    cells.append("—")
            data_rows.append(" & ".join(cells) + r" \\")

        # Assemble LaTeX
        lines = [
            r"\begin{table}[h]",
            r"  \centering",
            r"  \caption{FormatShield accuracy: direct / TTF ($\Delta$) by backend and task}",
            r"  \label{tab:accuracy_by_backend}",
            rf"  \begin{{tabular}}{{{col_spec}}}",
            r"    \toprule",
            "    " + header_row,
            r"    \midrule",
        ]
        for row in data_rows:
            lines.append("    " + row)
        lines += [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _result_to_row(result: BenchmarkResult) -> dict[str, Any]:
        """Convert a :class:`BenchmarkResult` to a flat CSV-serialisable dict."""
        row = result.to_dict()
        # Ensure failure_modes_detected is a plain comma-separated string
        fmd = row.get("failure_modes_detected", "")
        if isinstance(fmd, list):
            row["failure_modes_detected"] = ",".join(fmd)
        return row

    def load(self, input_path: Path) -> list[dict[str, Any]]:
        """
        Read a previously written CSV file back into a list of row dicts.

        Useful for inspecting exported data without loading a full
        :class:`BenchmarkResult` object graph.

        Parameters
        ----------
        input_path:
            Path to a CSV file written by :meth:`export` or
            :meth:`export_summary`.

        Returns
        -------
        list[dict]
            One dict per row, keyed by the CSV header columns.
        """
        input_path = Path(input_path)
        rows: list[dict[str, Any]] = []
        with input_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(dict(row))
        return rows

    def export_failure_modes(
        self,
        results: list[BenchmarkResult],
        output_path: Path,
    ) -> Path:
        """
        Write a CSV focused on failure-mode analysis.

        Each row represents a result that had at least one failure mode
        detected.  Columns: task, backend, model, failure_modes_detected,
        accuracy_delta, overhead_pct.

        Parameters
        ----------
        results:
            Full result list.  Entries with no failure modes are silently
            skipped.
        output_path:
            Destination file path.

        Returns
        -------
        Path
            The path to the written CSV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fields = [
            "task",
            "backend",
            "model",
            "failure_modes_detected",
            "accuracy_delta",
            "overhead_pct",
            "ttf_accuracy",
            "direct_accuracy",
        ]

        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                modes = r.failure_modes_detected
                if not modes:
                    continue
                writer.writerow(
                    {
                        "task": r.task,
                        "backend": r.backend,
                        "model": r.model,
                        "failure_modes_detected": (
                            ",".join(modes) if isinstance(modes, list) else str(modes)
                        ),
                        "accuracy_delta": r.accuracy_delta,
                        "overhead_pct": r.overhead_pct,
                        "ttf_accuracy": r.ttf_accuracy,
                        "direct_accuracy": r.direct_accuracy,
                    }
                )

        logger.info(
            "CSVExporter.export_failure_modes: wrote failure-mode CSV to %s",
            output_path,
        )
        return output_path

    def generate_summary_json(
        self,
        results: list[BenchmarkResult],
        output_path: Path,
    ) -> Path:
        """
        Write a JSON summary of all results, grouped by backend and task.

        Parameters
        ----------
        results:
            Full result list.
        output_path:
            Destination ``.json`` file path.

        Returns
        -------
        Path
            The path to the written JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        groups: dict[tuple[str, str], list[BenchmarkResult]] = defaultdict(list)
        for r in results:
            groups[(r.backend, r.task)].append(r)

        summary: dict[str, Any] = {
            "total_results": len(results),
            "backends": sorted({r.backend for r in results}),
            "tasks": sorted({r.task for r in results}),
            "groups": [],
        }

        for (backend, task), group in sorted(groups.items()):
            summary["groups"].append(
                {
                    "backend": backend,
                    "task": task,
                    "n": len(group),
                    "mean_accuracy_delta": round(_mean([r.accuracy_delta for r in group]), 4),
                    "mean_overhead_pct": round(_mean([r.overhead_pct for r in group]), 2),
                    "mean_complexity_score": round(_mean([r.complexity_score for r in group]), 4),
                }
            )

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        logger.info(
            "CSVExporter.generate_summary_json: wrote summary JSON to %s",
            output_path,
        )
        return output_path
