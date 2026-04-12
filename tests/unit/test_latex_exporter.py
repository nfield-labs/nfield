"""
Unit tests for formatshield.benchmark.exporters.latex_exporter.LaTeXExporter.

All tests use tmp_path for file I/O — no persistent file system state is
created or required.
"""

from __future__ import annotations

from pathlib import Path

from formatshield.benchmark.exporters.latex_exporter import LaTeXExporter
from formatshield.scorer.features import BenchmarkResult

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def make_result(
    task: str = "gsm",
    backend: str = "groq",
    accuracy_delta: float = 0.1,
    complexity_score: float = 0.7,
) -> BenchmarkResult:
    return BenchmarkResult(
        task=task,
        backend=backend,
        model="test-model",
        direct_accuracy=0.7,
        ttf_accuracy=0.7 + accuracy_delta,
        accuracy_delta=accuracy_delta,
        direct_latency_ms=100.0,
        ttf_latency_ms=120.0,
        overhead_pct=20.0,
        complexity_score=complexity_score,
        failure_modes_detected=[],
    )


def sample_results() -> list[BenchmarkResult]:
    """Return a mixed list of results for multiple backends/tasks."""
    return [
        make_result(task="gsm", backend="groq", accuracy_delta=0.1),
        make_result(task="gsm", backend="ollama", accuracy_delta=-0.05),
        make_result(task="medical_ner", backend="groq", accuracy_delta=0.2),
        make_result(task="template_fill", backend="vllm", accuracy_delta=0.0),
    ]


# ===========================================================================
# LaTeXExporter.export_main_table()
# ===========================================================================


class TestExportMainTable:
    def test_export_main_table_creates_file(self, tmp_path: Path) -> None:
        """export_main_table() must create the output .tex file."""
        exporter = LaTeXExporter()
        out = tmp_path / "main_table.tex"
        exporter.export_main_table(sample_results(), out)
        assert out.exists()

    def test_export_main_table_contains_booktabs(self, tmp_path: Path) -> None:
        """Output must contain all three booktabs rules."""
        exporter = LaTeXExporter()
        out = tmp_path / "main_table.tex"
        exporter.export_main_table(sample_results(), out)
        content = out.read_text(encoding="utf-8")
        assert r"\toprule" in content
        assert r"\midrule" in content
        assert r"\bottomrule" in content

    def test_export_main_table_contains_caption(self, tmp_path: Path) -> None:
        """Output must contain a \\caption command."""
        exporter = LaTeXExporter()
        out = tmp_path / "main_table.tex"
        exporter.export_main_table(sample_results(), out)
        content = out.read_text(encoding="utf-8")
        assert r"\caption" in content

    def test_export_main_table_positive_delta_bold(self, tmp_path: Path) -> None:
        """A positive accuracy_delta must render as \\textbf in the output."""
        exporter = LaTeXExporter()
        out = tmp_path / "main_table.tex"
        exporter.export_main_table([make_result(accuracy_delta=0.15)], out)
        content = out.read_text(encoding="utf-8")
        assert r"\textbf" in content

    def test_export_main_table_negative_delta_red(self, tmp_path: Path) -> None:
        """A negative accuracy_delta must render as \\textcolor{red} in the output."""
        exporter = LaTeXExporter()
        out = tmp_path / "main_table.tex"
        exporter.export_main_table([make_result(accuracy_delta=-0.1)], out)
        content = out.read_text(encoding="utf-8")
        assert r"\textcolor{red}" in content

    def test_export_main_table_empty_results(self, tmp_path: Path) -> None:
        """export_main_table([]) must write a 'No data' row and not raise."""
        exporter = LaTeXExporter()
        out = tmp_path / "main_empty.tex"
        exporter.export_main_table([], out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "No data" in content

    def test_export_main_table_contains_table_environment(self, tmp_path: Path) -> None:
        """Output must wrap content in a \\begin{table} ... \\end{table} block."""
        exporter = LaTeXExporter()
        out = tmp_path / "main_table.tex"
        exporter.export_main_table(sample_results(), out)
        content = out.read_text(encoding="utf-8")
        assert r"\begin{table}" in content
        assert r"\end{table}" in content

    def test_export_main_table_contains_tabular(self, tmp_path: Path) -> None:
        """Output must contain a \\begin{tabular} block."""
        exporter = LaTeXExporter()
        out = tmp_path / "main_table.tex"
        exporter.export_main_table(sample_results(), out)
        content = out.read_text(encoding="utf-8")
        assert r"\begin{tabular}" in content

    def test_export_main_table_zero_delta_plain(self, tmp_path: Path) -> None:
        """A zero accuracy_delta must not render as \\textbf or \\textcolor in data cells."""
        exporter = LaTeXExporter()
        out = tmp_path / "main_zero.tex"
        exporter.export_main_table([make_result(accuracy_delta=0.0)], out)
        content = out.read_text(encoding="utf-8")
        # Header row uses \textbf for column labels — check only the delta cell value.
        # _format_delta(0.0) returns "0.00" with no LaTeX markup.
        assert r"\textcolor{red}" not in content
        # The delta cell itself must be plain "0.00", not wrapped in \textbf{...}
        assert r"\textbf{+0" not in content
        assert r"\textbf{0" not in content

    def test_export_main_table_creates_parent_dirs(self, tmp_path: Path) -> None:
        """export_main_table() must create any missing parent directories."""
        exporter = LaTeXExporter()
        out = tmp_path / "deep" / "nested" / "main_table.tex"
        exporter.export_main_table([make_result()], out)
        assert out.exists()

    def test_export_main_table_contains_backend_name(self, tmp_path: Path) -> None:
        """The backend name must appear somewhere in the output."""
        exporter = LaTeXExporter()
        out = tmp_path / "main_table.tex"
        exporter.export_main_table([make_result(backend="groq")], out)
        content = out.read_text(encoding="utf-8")
        assert "groq" in content

    def test_export_main_table_contains_task_name(self, tmp_path: Path) -> None:
        """The task name must appear somewhere in the output."""
        exporter = LaTeXExporter()
        out = tmp_path / "main_table.tex"
        exporter.export_main_table([make_result(task="medical_ner")], out)
        content = out.read_text(encoding="utf-8")
        assert "medical" in content


# ===========================================================================
# LaTeXExporter.export_summary_stats()
# ===========================================================================


class TestExportSummaryStats:
    def test_export_summary_stats_creates_file(self, tmp_path: Path) -> None:
        """export_summary_stats() must create the output .tex file."""
        exporter = LaTeXExporter()
        out = tmp_path / "summary_stats.tex"
        exporter.export_summary_stats(sample_results(), out)
        assert out.exists()

    def test_export_summary_stats_contains_backend(self, tmp_path: Path) -> None:
        """The backend name must appear in the summary stats output."""
        exporter = LaTeXExporter()
        out = tmp_path / "summary_stats.tex"
        exporter.export_summary_stats([make_result(backend="vllm")], out)
        content = out.read_text(encoding="utf-8")
        assert "vllm" in content

    def test_export_summary_stats_empty_results(self, tmp_path: Path) -> None:
        """export_summary_stats([]) must write 'No data' and not raise."""
        exporter = LaTeXExporter()
        out = tmp_path / "summary_empty.tex"
        exporter.export_summary_stats([], out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "No data" in content

    def test_export_summary_stats_contains_booktabs(self, tmp_path: Path) -> None:
        """Summary stats output must contain booktabs rules."""
        exporter = LaTeXExporter()
        out = tmp_path / "summary_stats.tex"
        exporter.export_summary_stats(sample_results(), out)
        content = out.read_text(encoding="utf-8")
        assert r"\toprule" in content
        assert r"\bottomrule" in content

    def test_export_summary_stats_contains_caption(self, tmp_path: Path) -> None:
        """Summary stats output must contain a \\caption command."""
        exporter = LaTeXExporter()
        out = tmp_path / "summary_stats.tex"
        exporter.export_summary_stats(sample_results(), out)
        content = out.read_text(encoding="utf-8")
        assert r"\caption" in content

    def test_export_summary_stats_multiple_results_std(self, tmp_path: Path) -> None:
        """With multiple results per group, output must contain \\pm for std dev."""
        results = [
            make_result(task="gsm", backend="groq", accuracy_delta=0.1),
            make_result(task="gsm", backend="groq", accuracy_delta=0.3),
        ]
        exporter = LaTeXExporter()
        out = tmp_path / "summary_std.tex"
        exporter.export_summary_stats(results, out)
        content = out.read_text(encoding="utf-8")
        assert r"\pm" in content

    def test_export_summary_stats_single_result_no_std(self, tmp_path: Path) -> None:
        """With a single result per group, data cells must not contain \\pm (only header does)."""
        exporter = LaTeXExporter()
        out = tmp_path / "summary_single.tex"
        exporter.export_summary_stats([make_result(accuracy_delta=0.1)], out)
        content = out.read_text(encoding="utf-8")
        # The header row always contains \pm in its label text.
        # With n=1, data cell is just "0.10" (no \pm).
        # Verify the data row value "0.10" is present without a following \pm.
        assert "0.10 $\\pm$" not in content


# ===========================================================================
# LaTeXExporter.export_complexity_breakdown()
# ===========================================================================


class TestExportComplexityBreakdown:
    def test_export_complexity_breakdown_creates_file(self, tmp_path: Path) -> None:
        """export_complexity_breakdown() must create the output .tex file."""
        exporter = LaTeXExporter()
        out = tmp_path / "complexity.tex"
        exporter.export_complexity_breakdown(sample_results(), out)
        assert out.exists()

    def test_export_complexity_breakdown_three_buckets(self, tmp_path: Path) -> None:
        """Output must contain all three complexity bucket labels."""
        results = [
            make_result(complexity_score=0.1),   # [0.0, 0.3)
            make_result(complexity_score=0.45),  # [0.3, 0.6)
            make_result(complexity_score=0.8),   # [0.6, 1.0]
        ]
        exporter = LaTeXExporter()
        out = tmp_path / "complexity.tex"
        exporter.export_complexity_breakdown(results, out)
        content = out.read_text(encoding="utf-8")
        assert "[0.0, 0.3)" in content
        assert "[0.3, 0.6)" in content
        assert "[0.6, 1.0]" in content

    def test_export_complexity_breakdown_empty_results(self, tmp_path: Path) -> None:
        """export_complexity_breakdown([]) must write 'No data' and not raise."""
        exporter = LaTeXExporter()
        out = tmp_path / "complexity_empty.tex"
        exporter.export_complexity_breakdown([], out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "No data" in content

    def test_export_complexity_breakdown_contains_booktabs(self, tmp_path: Path) -> None:
        """Complexity breakdown output must contain booktabs rules."""
        exporter = LaTeXExporter()
        out = tmp_path / "complexity.tex"
        exporter.export_complexity_breakdown(sample_results(), out)
        content = out.read_text(encoding="utf-8")
        assert r"\toprule" in content
        assert r"\bottomrule" in content

    def test_export_complexity_breakdown_contains_caption(self, tmp_path: Path) -> None:
        """Complexity breakdown output must contain a \\caption command."""
        exporter = LaTeXExporter()
        out = tmp_path / "complexity.tex"
        exporter.export_complexity_breakdown(sample_results(), out)
        content = out.read_text(encoding="utf-8")
        assert r"\caption" in content

    def test_export_complexity_breakdown_dash_for_empty_bucket(
        self, tmp_path: Path
    ) -> None:
        """A bucket with no results must render '---' as its mean delta."""
        # All scores in [0.6, 1.0], so [0.0, 0.3) and [0.3, 0.6) are empty
        results = [make_result(complexity_score=0.9)]
        exporter = LaTeXExporter()
        out = tmp_path / "complexity_dash.tex"
        exporter.export_complexity_breakdown(results, out)
        content = out.read_text(encoding="utf-8")
        assert "---" in content


# ===========================================================================
# LaTeXExporter.generate_paper_tables()
# ===========================================================================


class TestGeneratePaperTables:
    def test_generate_paper_tables_returns_dict(self, tmp_path: Path) -> None:
        """generate_paper_tables() must return a dict."""
        exporter = LaTeXExporter()
        result = exporter.generate_paper_tables(sample_results(), tmp_path / "tables")
        assert isinstance(result, dict)

    def test_generate_paper_tables_dict_has_three_keys(self, tmp_path: Path) -> None:
        """The returned dict must have exactly 3 keys."""
        exporter = LaTeXExporter()
        result = exporter.generate_paper_tables(sample_results(), tmp_path / "tables")
        assert len(result) == 3

    def test_generate_paper_tables_dict_key_names(self, tmp_path: Path) -> None:
        """The returned dict must have the exact expected keys."""
        exporter = LaTeXExporter()
        result = exporter.generate_paper_tables(sample_results(), tmp_path / "tables")
        assert "main_table" in result
        assert "summary_stats" in result
        assert "complexity_breakdown" in result

    def test_generate_paper_tables_all_files_exist(self, tmp_path: Path) -> None:
        """All 3 output files returned in the dict must actually exist on disk."""
        exporter = LaTeXExporter()
        tables_dir = tmp_path / "tables"
        artifacts = exporter.generate_paper_tables(sample_results(), tables_dir)
        for key, path in artifacts.items():
            assert path.exists(), f"File for '{key}' does not exist: {path}"

    def test_generate_paper_tables_creates_dir(self, tmp_path: Path) -> None:
        """generate_paper_tables() must create the output directory if it is missing."""
        exporter = LaTeXExporter()
        tables_dir = tmp_path / "nonexistent" / "tables"
        assert not tables_dir.exists()
        exporter.generate_paper_tables(sample_results(), tables_dir)
        assert tables_dir.exists()

    def test_generate_paper_tables_values_are_paths(self, tmp_path: Path) -> None:
        """The values in the returned dict must all be Path instances."""
        exporter = LaTeXExporter()
        result = exporter.generate_paper_tables(sample_results(), tmp_path / "tables")
        for path in result.values():
            assert isinstance(path, Path)

    def test_generate_paper_tables_empty_results_does_not_raise(
        self, tmp_path: Path
    ) -> None:
        """generate_paper_tables([]) must not raise and must still create files."""
        exporter = LaTeXExporter()
        tables_dir = tmp_path / "empty_tables"
        artifacts = exporter.generate_paper_tables([], tables_dir)
        for key, path in artifacts.items():
            assert path.exists(), f"File for '{key}' does not exist: {path}"


# ===========================================================================
# LaTeXExporter._escape_latex()
# ===========================================================================


class TestEscapeLatex:
    def test_escape_latex_ampersand(self) -> None:
        """& must be escaped to \\&."""
        exporter = LaTeXExporter()
        assert exporter._escape_latex("a & b") == r"a \& b"

    def test_escape_latex_percent(self) -> None:
        """% must be escaped to \\%."""
        exporter = LaTeXExporter()
        assert exporter._escape_latex("50%") == r"50\%"

    def test_escape_latex_dollar(self) -> None:
        """$ must be escaped to \\$."""
        exporter = LaTeXExporter()
        assert exporter._escape_latex("$100") == r"\$100"

    def test_escape_latex_underscore(self) -> None:
        """_ must be escaped to \\_."""
        exporter = LaTeXExporter()
        assert exporter._escape_latex("snake_case") == r"snake\_case"

    def test_escape_latex_backslash(self) -> None:
        r"""\ must be escaped to \textbackslash\{\} (braces are subsequently escaped)."""
        exporter = LaTeXExporter()
        # The implementation replaces \ with \textbackslash{} first, then later
        # escapes { -> \{ and } -> \}, so the final form is \textbackslash\{\}.
        assert exporter._escape_latex("a\\b") == r"a\textbackslash\{\}b"

    def test_escape_latex_hash(self) -> None:
        """# must be escaped to \\#."""
        exporter = LaTeXExporter()
        assert exporter._escape_latex("#tag") == r"\#tag"

    def test_escape_latex_brace_open(self) -> None:
        """{ must be escaped to \\{."""
        exporter = LaTeXExporter()
        assert exporter._escape_latex("{value}") == r"\{value\}"

    def test_escape_latex_tilde(self) -> None:
        """~ must be escaped to \\textasciitilde{}."""
        exporter = LaTeXExporter()
        assert exporter._escape_latex("a~b") == r"a\textasciitilde{}b"

    def test_escape_latex_caret(self) -> None:
        """^ must be escaped to \\textasciicircum{}."""
        exporter = LaTeXExporter()
        assert exporter._escape_latex("x^2") == r"x\textasciicircum{}2"

    def test_escape_latex_plain_text_unchanged(self) -> None:
        """Plain ASCII text with no special characters must be returned unchanged."""
        exporter = LaTeXExporter()
        assert exporter._escape_latex("Hello World") == "Hello World"

    def test_escape_latex_empty_string(self) -> None:
        """An empty string must be returned unchanged."""
        exporter = LaTeXExporter()
        assert exporter._escape_latex("") == ""

    def test_escape_latex_backslash_not_double_escaped(self) -> None:
        r"""Backslash replacement must not cause double-escaping of the & that follows."""
        exporter = LaTeXExporter()
        # \ -> \textbackslash{}, then { -> \{, } -> \}, then & -> \&
        # Final result: \textbackslash\{\}\&
        result = exporter._escape_latex("\\&")
        assert result == r"\textbackslash\{\}\&"
