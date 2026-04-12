"""
Unit tests for formatshield.benchmark.exporters.csv_exporter.CSVExporter.

All tests use tmp_path for file I/O — no persistent file system state is
created or required.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from formatshield.benchmark.exporters.csv_exporter import CSVExporter
from formatshield.scorer.features import BenchmarkResult

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_result(
    *,
    task: str = "gsm_symbolic",
    backend: str = "groq",
    model: str = "groq/llama3",
    direct_accuracy: float = 0.6,
    ttf_accuracy: float = 0.8,
    direct_latency_ms: float = 200.0,
    ttf_latency_ms: float = 500.0,
    overhead_pct: float = 150.0,
    complexity_score: float = 0.82,
    failure_modes: list[str] | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        task=task,
        backend=backend,
        model=model,
        direct_accuracy=direct_accuracy,
        ttf_accuracy=ttf_accuracy,
        accuracy_delta=ttf_accuracy - direct_accuracy,
        direct_latency_ms=direct_latency_ms,
        ttf_latency_ms=ttf_latency_ms,
        overhead_pct=overhead_pct,
        complexity_score=complexity_score,
        failure_modes_detected=failure_modes or [],
    )


def _sample_results() -> list[BenchmarkResult]:
    """Return a mixed list of results for multiple backends/tasks."""
    return [
        _make_result(task="gsm_symbolic", backend="groq"),
        _make_result(task="gsm_symbolic", backend="ollama"),
        _make_result(task="medical_ner", backend="groq", failure_modes=["high_overhead_low_gain"]),
        _make_result(task="template_fill", backend="groq", direct_accuracy=0.9, ttf_accuracy=0.85),
    ]


# ===========================================================================
# CSVExporter.export()
# ===========================================================================


class TestCSVExporterExport:
    def test_export_creates_file(self, tmp_path: Path) -> None:
        """export() must create the output CSV file."""
        exporter = CSVExporter()
        out = tmp_path / "raw.csv"
        exporter.export(_sample_results(), out)
        assert out.exists()

    def test_export_returns_path(self, tmp_path: Path) -> None:
        """export() must return the path to the created file."""
        exporter = CSVExporter()
        out = tmp_path / "raw.csv"
        result_path = exporter.export(_sample_results(), out)
        assert result_path == out

    def test_export_has_correct_headers(self, tmp_path: Path) -> None:
        """The exported CSV must contain all expected header columns."""
        exporter = CSVExporter()
        out = tmp_path / "raw.csv"
        exporter.export(_sample_results(), out)
        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
        expected = {
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
        }
        assert expected.issubset(set(headers))

    def test_export_row_count_matches_results(self, tmp_path: Path) -> None:
        """The CSV must contain one data row per BenchmarkResult."""
        results = _sample_results()
        exporter = CSVExporter()
        out = tmp_path / "raw.csv"
        exporter.export(results, out)
        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == len(results)

    def test_export_values_are_correct(self, tmp_path: Path) -> None:
        """Exported values must match the source BenchmarkResult."""
        r = _make_result(task="medical_ner", backend="vllm", direct_accuracy=0.75)
        exporter = CSVExporter()
        out = tmp_path / "single.csv"
        exporter.export([r], out)
        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert rows[0]["task"] == "medical_ner"
        assert rows[0]["backend"] == "vllm"
        assert float(rows[0]["direct_accuracy"]) == pytest.approx(0.75)

    def test_export_failure_modes_serialised_as_string(self, tmp_path: Path) -> None:
        """failure_modes_detected must be serialised as a comma-joined string."""
        r = _make_result(failure_modes=["mode_a", "mode_b"])
        exporter = CSVExporter()
        out = tmp_path / "fm.csv"
        exporter.export([r], out)
        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert "mode_a" in rows[0]["failure_modes_detected"]
        assert "mode_b" in rows[0]["failure_modes_detected"]

    def test_export_empty_results_creates_header_only_csv(self, tmp_path: Path) -> None:
        """export([]) must create a valid CSV with just the header row."""
        exporter = CSVExporter()
        out = tmp_path / "empty.csv"
        exporter.export([], out)
        assert out.exists()
        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert rows == []

    def test_export_creates_parent_dirs(self, tmp_path: Path) -> None:
        """export() must create missing parent directories."""
        exporter = CSVExporter()
        out = tmp_path / "deep" / "nested" / "raw.csv"
        exporter.export([_make_result()], out)
        assert out.exists()


# ===========================================================================
# CSVExporter.export_summary()
# ===========================================================================


class TestCSVExporterExportSummary:
    def test_export_summary_creates_file(self, tmp_path: Path) -> None:
        """export_summary() must create the output CSV file."""
        exporter = CSVExporter()
        out = tmp_path / "summary.csv"
        exporter.export_summary(_sample_results(), out)
        assert out.exists()

    def test_export_summary_returns_path(self, tmp_path: Path) -> None:
        """export_summary() must return the path to the created file."""
        exporter = CSVExporter()
        out = tmp_path / "summary.csv"
        result_path = exporter.export_summary(_sample_results(), out)
        assert result_path == out

    def test_export_summary_aggregates_by_backend_and_task(self, tmp_path: Path) -> None:
        """export_summary() must produce one row per (backend, task) pair."""
        results = [
            _make_result(task="gsm_symbolic", backend="groq"),
            _make_result(task="gsm_symbolic", backend="groq"),
            _make_result(task="medical_ner", backend="groq"),
        ]
        exporter = CSVExporter()
        out = tmp_path / "summary.csv"
        exporter.export_summary(results, out)
        with out.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2  # (groq, gsm_symbolic) and (groq, medical_ner)

    def test_export_summary_has_correct_headers(self, tmp_path: Path) -> None:
        """Summary CSV must include all summary-specific columns."""
        exporter = CSVExporter()
        out = tmp_path / "summary.csv"
        exporter.export_summary(_sample_results(), out)
        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
        expected = {
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
        }
        assert expected.issubset(set(headers))

    def test_export_summary_n_problems_correct(self, tmp_path: Path) -> None:
        """n_problems must reflect the number of results in each group."""
        results = [_make_result() for _ in range(4)]
        exporter = CSVExporter()
        out = tmp_path / "summary.csv"
        exporter.export_summary(results, out)
        with out.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["n_problems"] == "4"

    def test_export_summary_empty_results_creates_header_only(self, tmp_path: Path) -> None:
        """export_summary([]) must not raise and must create a valid CSV."""
        exporter = CSVExporter()
        out = tmp_path / "empty_summary.csv"
        exporter.export_summary([], out)
        assert out.exists()
        with out.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows == []

    def test_export_summary_mean_accuracy_delta(self, tmp_path: Path) -> None:
        """mean_accuracy_delta must be the arithmetic mean of accuracy_deltas."""
        results = [
            _make_result(direct_accuracy=0.5, ttf_accuracy=0.7),  # delta=0.2
            _make_result(direct_accuracy=0.6, ttf_accuracy=0.8),  # delta=0.2
        ]
        exporter = CSVExporter()
        out = tmp_path / "summary_delta.csv"
        exporter.export_summary(results, out)
        with out.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert float(rows[0]["mean_accuracy_delta"]) == pytest.approx(0.2, abs=0.01)


# ===========================================================================
# CSVExporter.generate_latex_table()
# ===========================================================================


class TestCSVExporterGenerateLatexTable:
    def test_generate_latex_table_returns_string(self) -> None:
        """generate_latex_table() must return a string."""
        exporter = CSVExporter()
        result = exporter.generate_latex_table(_sample_results())
        assert isinstance(result, str)

    def test_generate_latex_table_contains_tabular(self) -> None:
        """The returned LaTeX must contain 'tabular'."""
        exporter = CSVExporter()
        result = exporter.generate_latex_table(_sample_results())
        assert "tabular" in result

    def test_generate_latex_table_contains_begin_table(self) -> None:
        """The returned LaTeX must contain a table environment open."""
        exporter = CSVExporter()
        result = exporter.generate_latex_table(_sample_results())
        assert r"\begin{table}" in result

    def test_generate_latex_table_contains_end_table(self) -> None:
        """The returned LaTeX must contain a table environment close."""
        exporter = CSVExporter()
        result = exporter.generate_latex_table(_sample_results())
        assert r"\end{table}" in result

    def test_generate_latex_table_contains_backend_names(self) -> None:
        """Backend names must appear in the generated LaTeX."""
        exporter = CSVExporter()
        result = exporter.generate_latex_table(_sample_results())
        assert "groq" in result

    def test_generate_latex_table_contains_task_names(self) -> None:
        """Task names must appear in the generated LaTeX."""
        exporter = CSVExporter()
        result = exporter.generate_latex_table(_sample_results())
        assert "gsm" in result or "medical" in result

    def test_generate_latex_table_empty_results_does_not_raise(self) -> None:
        """generate_latex_table([]) must not raise."""
        exporter = CSVExporter()
        result = exporter.generate_latex_table([])
        assert isinstance(result, str)

    def test_generate_latex_table_contains_caption(self) -> None:
        """The LaTeX output must contain a \\caption command."""
        exporter = CSVExporter()
        result = exporter.generate_latex_table(_sample_results())
        assert r"\caption" in result

    def test_generate_latex_table_single_result(self) -> None:
        """generate_latex_table() must work with a single-result input."""
        exporter = CSVExporter()
        result = exporter.generate_latex_table([_make_result()])
        assert "tabular" in result


# ===========================================================================
# CSVExporter.export_failure_modes()
# ===========================================================================


class TestCSVExporterExportFailureModes:
    def test_export_failure_modes_creates_file(self, tmp_path: Path) -> None:
        """export_failure_modes() must create the output CSV file."""
        exporter = CSVExporter()
        out = tmp_path / "failures.csv"
        results = [_make_result(failure_modes=["ttf_accuracy_regression"])]
        exporter.export_failure_modes(results, out)
        assert out.exists()

    def test_export_failure_modes_returns_path(self, tmp_path: Path) -> None:
        """export_failure_modes() must return the path to the output file."""
        exporter = CSVExporter()
        out = tmp_path / "failures.csv"
        result_path = exporter.export_failure_modes([_make_result(failure_modes=["mode"])], out)
        assert result_path == out

    def test_export_failure_modes_skips_clean_results(self, tmp_path: Path) -> None:
        """Results with no failure modes must not appear in the output CSV."""
        results = [
            _make_result(failure_modes=[]),
            _make_result(failure_modes=["ttf_accuracy_regression"]),
        ]
        exporter = CSVExporter()
        out = tmp_path / "failures.csv"
        exporter.export_failure_modes(results, out)
        with out.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1

    def test_export_failure_modes_has_correct_columns(self, tmp_path: Path) -> None:
        """The failure-modes CSV must include the expected columns."""
        exporter = CSVExporter()
        out = tmp_path / "failures.csv"
        exporter.export_failure_modes([_make_result(failure_modes=["high_overhead_low_gain"])], out)
        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
        expected = {
            "task",
            "backend",
            "model",
            "failure_modes_detected",
            "accuracy_delta",
            "overhead_pct",
            "ttf_accuracy",
            "direct_accuracy",
        }
        assert expected.issubset(set(headers))

    def test_export_failure_modes_empty_results_does_not_raise(self, tmp_path: Path) -> None:
        """export_failure_modes([]) must not raise."""
        exporter = CSVExporter()
        out = tmp_path / "empty_failures.csv"
        exporter.export_failure_modes([], out)
        assert out.exists()

    def test_export_failure_modes_all_clean_produces_empty_rows(self, tmp_path: Path) -> None:
        """When all results are clean, the CSV must have zero data rows."""
        results = [_make_result(failure_modes=[]) for _ in range(3)]
        exporter = CSVExporter()
        out = tmp_path / "all_clean.csv"
        exporter.export_failure_modes(results, out)
        with out.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows == []

    def test_export_failure_modes_mode_label_in_output(self, tmp_path: Path) -> None:
        """The failure mode label must appear in the output CSV."""
        exporter = CSVExporter()
        out = tmp_path / "mode_label.csv"
        exporter.export_failure_modes(
            [_make_result(failure_modes=["unnecessary_ttf_overhead"])], out
        )
        with out.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert "unnecessary_ttf_overhead" in rows[0]["failure_modes_detected"]


# ===========================================================================
# CSVExporter.generate_summary_json()
# ===========================================================================


class TestCSVExporterGenerateSummaryJson:
    def test_generate_summary_json_creates_file(self, tmp_path: Path) -> None:
        """generate_summary_json() must create the output JSON file."""
        exporter = CSVExporter()
        out = tmp_path / "summary.json"
        exporter.generate_summary_json(_sample_results(), out)
        assert out.exists()

    def test_generate_summary_json_returns_path(self, tmp_path: Path) -> None:
        """generate_summary_json() must return the path to the created file."""
        exporter = CSVExporter()
        out = tmp_path / "summary.json"
        result_path = exporter.generate_summary_json(_sample_results(), out)
        assert result_path == out

    def test_generate_summary_json_is_valid_json(self, tmp_path: Path) -> None:
        """The generated file must contain valid JSON."""
        exporter = CSVExporter()
        out = tmp_path / "summary.json"
        exporter.generate_summary_json(_sample_results(), out)
        with out.open(encoding="utf-8") as fh:
            data = json.load(fh)
        assert data is not None

    def test_generate_summary_json_total_results_correct(self, tmp_path: Path) -> None:
        """total_results in the JSON must match the number of input results."""
        results = _sample_results()
        exporter = CSVExporter()
        out = tmp_path / "summary.json"
        exporter.generate_summary_json(results, out)
        with out.open(encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["total_results"] == len(results)

    def test_generate_summary_json_backends_list(self, tmp_path: Path) -> None:
        """JSON must contain a 'backends' list with all unique backends."""
        exporter = CSVExporter()
        out = tmp_path / "summary.json"
        exporter.generate_summary_json(_sample_results(), out)
        with out.open(encoding="utf-8") as fh:
            data = json.load(fh)
        assert "backends" in data
        assert "groq" in data["backends"]

    def test_generate_summary_json_tasks_list(self, tmp_path: Path) -> None:
        """JSON must contain a 'tasks' list with all unique task names."""
        exporter = CSVExporter()
        out = tmp_path / "summary.json"
        exporter.generate_summary_json(_sample_results(), out)
        with out.open(encoding="utf-8") as fh:
            data = json.load(fh)
        assert "tasks" in data
        assert "gsm_symbolic" in data["tasks"]

    def test_generate_summary_json_groups_list(self, tmp_path: Path) -> None:
        """JSON must contain a 'groups' list."""
        exporter = CSVExporter()
        out = tmp_path / "summary.json"
        exporter.generate_summary_json(_sample_results(), out)
        with out.open(encoding="utf-8") as fh:
            data = json.load(fh)
        assert "groups" in data
        assert isinstance(data["groups"], list)

    def test_generate_summary_json_empty_results_does_not_raise(self, tmp_path: Path) -> None:
        """generate_summary_json([]) must not raise."""
        exporter = CSVExporter()
        out = tmp_path / "empty_summary.json"
        exporter.generate_summary_json([], out)
        assert out.exists()

    def test_generate_summary_json_empty_results_valid_json(self, tmp_path: Path) -> None:
        """Even with an empty input, the JSON file must be valid."""
        exporter = CSVExporter()
        out = tmp_path / "empty.json"
        exporter.generate_summary_json([], out)
        with out.open(encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["total_results"] == 0

    def test_generate_summary_json_group_has_mean_accuracy_delta(self, tmp_path: Path) -> None:
        """Each group entry in the JSON must include mean_accuracy_delta."""
        exporter = CSVExporter()
        out = tmp_path / "groups.json"
        exporter.generate_summary_json(_sample_results(), out)
        with out.open(encoding="utf-8") as fh:
            data = json.load(fh)
        for group in data["groups"]:
            assert "mean_accuracy_delta" in group


# ===========================================================================
# CSVExporter.load()
# ===========================================================================


class TestCSVExporterLoad:
    def test_load_reads_exported_rows(self, tmp_path: Path) -> None:
        """load() must read back the rows written by export()."""
        results = [_make_result()]
        exporter = CSVExporter()
        out = tmp_path / "roundtrip.csv"
        exporter.export(results, out)
        loaded = exporter.load(out)
        assert len(loaded) == 1

    def test_load_returns_list_of_dicts(self, tmp_path: Path) -> None:
        """load() must return a list of dicts."""
        exporter = CSVExporter()
        out = tmp_path / "load_test.csv"
        exporter.export([_make_result()], out)
        loaded = exporter.load(out)
        assert isinstance(loaded, list)
        assert all(isinstance(row, dict) for row in loaded)

    def test_load_task_value_preserved(self, tmp_path: Path) -> None:
        """load() must preserve the 'task' field value."""
        r = _make_result(task="medical_ner")
        exporter = CSVExporter()
        out = tmp_path / "task_test.csv"
        exporter.export([r], out)
        loaded = exporter.load(out)
        assert loaded[0]["task"] == "medical_ner"
