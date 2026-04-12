"""
LaTeXExporter — writes FormatShield benchmark results to LaTeX table format.

All I/O uses the Python standard library only; no external dependencies
(pandas, numpy, matplotlib) are required for this module.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from formatshield.scorer.features import BenchmarkResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Complexity score bucket definitions
# ---------------------------------------------------------------------------

_COMPLEXITY_BUCKETS: list[tuple[str, float, float]] = [
    ("[0.0, 0.3)", 0.0, 0.3),
    ("[0.3, 0.6)", 0.3, 0.6),
    ("[0.6, 1.0]", 0.6, 1.0),
]


class LaTeXExporter:
    """Export benchmark results to LaTeX table format for academic papers.

    All methods are stateless and can be called on a single instance
    multiple times with different result sets.

    Example::

        exporter = LaTeXExporter()
        tables = exporter.generate_paper_tables(results, Path("out/tables"))
        print(tables)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_main_table(
        self,
        results: list[BenchmarkResult],
        output_path: Path,
    ) -> None:
        """Write the main benchmark comparison table (Table 1 in the paper).

        Columns: Task | Backend | Model | Direct Acc | TTF Acc | Δ Accuracy |
        Overhead %.  Rows are grouped first by task, then by backend.  A
        positive Δ is rendered in bold; a negative Δ is rendered in red using
        ``\\textcolor{red}{...}``.

        The output is a complete LaTeX ``table`` environment using the
        ``booktabs`` package (``\\toprule``, ``\\midrule``, ``\\bottomrule``).

        Parameters
        ----------
        results:
            List of :class:`~formatshield.scorer.features.BenchmarkResult`
            instances to tabulate.
        output_path:
            Destination ``.tex`` file path.  Parent directories are created
            automatically.

        Returns
        -------
        None
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        col_spec = "llllllr"
        header_cols = [
            r"\textbf{Task}",
            r"\textbf{Backend}",
            r"\textbf{Model}",
            r"\textbf{Direct Acc}",
            r"\textbf{TTF Acc}",
            r"\textbf{$\Delta$ Accuracy}",
            r"\textbf{Overhead \%}",
        ]
        header_row = " & ".join(header_cols) + r" \\"

        data_rows: list[str] = []

        if not results:
            no_data_cols = [r"\multicolumn{7}{c}{No data}"]
            data_rows.append(" & ".join(no_data_cols) + r" \\")
        else:
            # Group: task -> backend -> list[BenchmarkResult]
            grouped: dict[str, dict[str, list[BenchmarkResult]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for r in results:
                grouped[r.task][r.backend].append(r)

            first_task = True
            for task in sorted(grouped):
                if not first_task:
                    data_rows.append(r"    \midrule")
                first_task = False

                first_backend = True
                for backend in sorted(grouped[task]):
                    for r in grouped[task][backend]:
                        task_cell = self._escape_latex(r.task) if first_backend else ""
                        delta_str = self._format_delta(r.accuracy_delta)
                        row_cols = [
                            task_cell,
                            self._escape_latex(r.backend),
                            self._escape_latex(r.model),
                            f"{r.direct_accuracy:.2f}",
                            f"{r.ttf_accuracy:.2f}",
                            delta_str,
                            f"{r.overhead_pct:.2f}",
                        ]
                        data_rows.append(" & ".join(row_cols) + r" \\")
                    first_backend = False

        lines = [
            r"\begin{table}[h]",
            r"  \centering",
            r"  \caption{FormatShield Benchmark Results: Direct vs.\ TTF Accuracy}",
            r"  \label{tab:main_results}",
            rf"  \begin{{tabular}}{{{col_spec}}}",
            r"    \toprule",
            "    " + header_row,
            r"    \midrule",
        ]
        for row in data_rows:
            if row.startswith(r"    \midrule"):
                lines.append(row)
            else:
                lines.append("    " + row)
        lines += [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]

        content = "\n".join(lines) + "\n"
        output_path.write_text(content, encoding="utf-8")
        logger.info("LaTeXExporter.export_main_table: wrote %s", output_path)

    def export_summary_stats(
        self,
        results: list[BenchmarkResult],
        output_path: Path,
    ) -> None:
        """Write a summary statistics table aggregated by (backend, task).

        Each row represents one backend × task combination showing
        mean ± std for ``accuracy_delta`` and ``overhead_pct`` across all
        results in that group.

        Parameters
        ----------
        results:
            List of :class:`~formatshield.scorer.features.BenchmarkResult`
            instances to aggregate.
        output_path:
            Destination ``.tex`` file path.  Parent directories are created
            automatically.

        Returns
        -------
        None
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        col_spec = "llccc"
        header_cols = [
            r"\textbf{Backend}",
            r"\textbf{Task}",
            r"\textbf{N}",
            r"\textbf{$\Delta$ Acc (mean $\pm$ std)}",
            r"\textbf{Overhead \% (mean $\pm$ std)}",
        ]
        header_row = " & ".join(header_cols) + r" \\"

        data_rows: list[str] = []

        if not results:
            data_rows.append(r"\multicolumn{5}{c}{No data} \\")
        else:
            groups: dict[tuple[str, str], list[BenchmarkResult]] = defaultdict(list)
            for r in results:
                groups[(r.backend, r.task)].append(r)

            for (backend, task), group in sorted(groups.items()):
                n = len(group)
                deltas = [r.accuracy_delta for r in group]
                overheads = [r.overhead_pct for r in group]

                mean_delta = mean(deltas)
                mean_overhead = mean(overheads)

                if n > 1:
                    std_delta = stdev(deltas)
                    std_overhead = stdev(overheads)
                    delta_cell = (
                        f"{mean_delta:.2f} $\\pm$ {std_delta:.2f}"
                    )
                    overhead_cell = (
                        f"{mean_overhead:.2f} $\\pm$ {std_overhead:.2f}"
                    )
                else:
                    delta_cell = f"{mean_delta:.2f}"
                    overhead_cell = f"{mean_overhead:.2f}"

                row_cols = [
                    self._escape_latex(backend),
                    self._escape_latex(task),
                    str(n),
                    delta_cell,
                    overhead_cell,
                ]
                data_rows.append(" & ".join(row_cols) + r" \\")

        lines = [
            r"\begin{table}[h]",
            r"  \centering",
            r"  \caption{Summary Statistics}",
            r"  \label{tab:summary}",
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

        content = "\n".join(lines) + "\n"
        output_path.write_text(content, encoding="utf-8")
        logger.info("LaTeXExporter.export_summary_stats: wrote %s", output_path)

    def export_complexity_breakdown(
        self,
        results: list[BenchmarkResult],
        output_path: Path,
    ) -> None:
        """Write a complexity-score breakdown table.

        Results are grouped into three complexity buckets:
        ``[0.0, 0.3)``, ``[0.3, 0.6)``, ``[0.6, 1.0]``.  Each row shows the
        bucket label, the count of results in that bucket, and the mean
        ``accuracy_delta`` for the bucket.

        Parameters
        ----------
        results:
            List of :class:`~formatshield.scorer.features.BenchmarkResult`
            instances to group.
        output_path:
            Destination ``.tex`` file path.  Parent directories are created
            automatically.

        Returns
        -------
        None
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        col_spec = "lcc"
        header_cols = [
            r"\textbf{Complexity Bucket}",
            r"\textbf{Count}",
            r"\textbf{Mean $\Delta$ Accuracy}",
        ]
        header_row = " & ".join(header_cols) + r" \\"

        # Assign results to buckets
        buckets: dict[str, list[float]] = {label: [] for label, _, _ in _COMPLEXITY_BUCKETS}

        for r in results:
            score = r.complexity_score
            for label, low, high in _COMPLEXITY_BUCKETS:
                if low <= score < high or (high == 1.0 and score == 1.0):
                    buckets[label].append(r.accuracy_delta)
                    break

        data_rows: list[str] = []

        if not results:
            data_rows.append(r"\multicolumn{3}{c}{No data} \\")
        else:
            for label, _, _ in _COMPLEXITY_BUCKETS:
                deltas = buckets[label]
                count = len(deltas)
                mean_delta_str = f"{mean(deltas):.2f}" if deltas else "---"
                row_cols = [
                    self._escape_latex(label),
                    str(count),
                    mean_delta_str,
                ]
                data_rows.append(" & ".join(row_cols) + r" \\")

        lines = [
            r"\begin{table}[h]",
            r"  \centering",
            r"  \caption{Accuracy Delta by Complexity Score Bucket}",
            r"  \label{tab:complexity_breakdown}",
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

        content = "\n".join(lines) + "\n"
        output_path.write_text(content, encoding="utf-8")
        logger.info(
            "LaTeXExporter.export_complexity_breakdown: wrote %s", output_path
        )

    def generate_paper_tables(
        self,
        results: list[BenchmarkResult],
        tables_dir: Path,
    ) -> dict[str, Path]:
        """Generate all three paper tables and write them to *tables_dir*.

        Creates the output directory (and any missing parents) if it does not
        already exist.

        Parameters
        ----------
        results:
            Full list of :class:`~formatshield.scorer.features.BenchmarkResult`
            instances from the benchmark harness.
        tables_dir:
            Directory where the three ``.tex`` files are written.

        Returns
        -------
        dict[str, Path]
            Mapping with keys ``"main_table"``, ``"summary_stats"``, and
            ``"complexity_breakdown"``, each pointing to the corresponding
            written file path.
        """
        tables_dir = Path(tables_dir)
        tables_dir.mkdir(parents=True, exist_ok=True)

        main_path = tables_dir / "main_table.tex"
        summary_path = tables_dir / "summary_stats.tex"
        complexity_path = tables_dir / "complexity_breakdown.tex"

        self.export_main_table(results, main_path)
        self.export_summary_stats(results, summary_path)
        self.export_complexity_breakdown(results, complexity_path)

        artifacts: dict[str, Path] = {
            "main_table": main_path,
            "summary_stats": summary_path,
            "complexity_breakdown": complexity_path,
        }
        logger.info(
            "LaTeXExporter.generate_paper_tables: wrote %d tables to %s",
            len(artifacts),
            tables_dir,
        )
        return artifacts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _escape_latex(text: str) -> str:
        """Escape special LaTeX characters in *text*.

        The following characters are escaped: ``& % $ # _ { } ~ ^ \\``

        Parameters
        ----------
        text:
            Raw string that may contain LaTeX-special characters.

        Returns
        -------
        str
            A copy of *text* with all special characters escaped so the
            string can be safely embedded in a LaTeX document.
        """
        # Order matters: backslash must be replaced first to avoid
        # double-escaping the backslashes introduced by subsequent replacements.
        replacements: list[tuple[str, str]] = [
            ("\\", r"\textbackslash{}"),
            ("&", r"\&"),
            ("%", r"\%"),
            ("$", r"\$"),
            ("#", r"\#"),
            ("_", r"\_"),
            ("{", r"\{"),
            ("}", r"\}"),
            ("~", r"\textasciitilde{}"),
            ("^", r"\textasciicircum{}"),
        ]
        for char, replacement in replacements:
            text = text.replace(char, replacement)
        return text

    @staticmethod
    def _format_delta(delta: float) -> str:
        """Format an accuracy delta value with LaTeX styling.

        Positive deltas (TTF helps) are rendered bold.
        Negative deltas (TTF hurts) are rendered in red via
        ``\\textcolor{red}{...}``.
        Zero is rendered plainly.

        Parameters
        ----------
        delta:
            The ``accuracy_delta`` value to format.

        Returns
        -------
        str
            A LaTeX snippet ready for inclusion in a table cell.
        """
        sign = "+" if delta > 0 else ""
        formatted = f"{sign}{delta:.2f}"
        if delta > 0:
            return rf"\textbf{{{formatted}}}"
        if delta < 0:
            return rf"\textcolor{{red}}{{{formatted}}}"
        return formatted
