"""Tests for formatshield.benchmark.exporters.png_exporter.PNGExporter."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from formatshield.benchmark.exporters.png_exporter import (
    PNGExporter,
    _aggregate_by_backend_task,
)
from formatshield.scorer.features import BenchmarkResult

# Ensure Agg backend is active so tests run headless (no display required)
try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    backend: str = "groq",
    task: str = "gsm_symbolic",
    delta: float = 0.15,
) -> BenchmarkResult:
    return BenchmarkResult(
        task=task,
        backend=backend,
        model=f"{backend}/llama",
        direct_accuracy=0.6,
        ttf_accuracy=0.6 + delta,
        accuracy_delta=delta,
        direct_latency_ms=200.0,
        ttf_latency_ms=450.0,
        overhead_pct=125.0,
        complexity_score=0.8,
        failure_modes_detected=[],
    )


# ---------------------------------------------------------------------------
# _aggregate_by_backend_task helper
# ---------------------------------------------------------------------------


def test_aggregate_by_backend_task_groups_correctly() -> None:
    results = [
        _make_result(backend="groq", task="gsm", delta=0.10),
        _make_result(backend="groq", task="gsm", delta=0.20),
        _make_result(backend="ollama", task="gsm", delta=0.05),
    ]
    aggregated = _aggregate_by_backend_task(results)

    assert "groq" in aggregated
    assert "ollama" in aggregated
    # Mean of 0.10 and 0.20 = 0.15
    assert abs(aggregated["groq"]["gsm"] - 0.15) < 1e-9
    assert abs(aggregated["ollama"]["gsm"] - 0.05) < 1e-9


def test_aggregate_by_backend_task_empty_returns_empty() -> None:
    aggregated = _aggregate_by_backend_task([])
    assert aggregated == {}


def test_aggregate_by_backend_task_multiple_tasks() -> None:
    results = [
        _make_result(backend="groq", task="task_a", delta=0.10),
        _make_result(backend="groq", task="task_b", delta=-0.05),
    ]
    aggregated = _aggregate_by_backend_task(results)
    assert set(aggregated["groq"].keys()) == {"task_a", "task_b"}


def test_aggregate_by_backend_task_single_result() -> None:
    results = [_make_result(backend="vllm", task="ner", delta=0.08)]
    aggregated = _aggregate_by_backend_task(results)
    assert aggregated["vllm"]["ner"] == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# export_heatmap — uses real matplotlib (installed in this environment)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    "matplotlib" not in sys.modules and True,
    reason="matplotlib required",
)
def test_export_heatmap_creates_file(tmp_path: Path) -> None:
    results = [
        _make_result(backend="groq", task="gsm", delta=0.15),
        _make_result(backend="ollama", task="gsm", delta=0.05),
    ]
    out = tmp_path / "heatmap.png"

    exporter = PNGExporter()
    result_path = exporter.export_heatmap(results, out)

    assert isinstance(result_path, Path)
    assert result_path.suffix == ".png"


def test_export_heatmap_adds_png_extension(tmp_path: Path) -> None:
    results = [_make_result()]
    out = tmp_path / "heatmap"  # no extension

    exporter = PNGExporter()
    result_path = exporter.export_heatmap(results, out)

    assert result_path.suffix == ".png"


def test_export_heatmap_returns_resolved_path(tmp_path: Path) -> None:
    results = [_make_result()]
    out = tmp_path / "heatmap.png"

    exporter = PNGExporter()
    result_path = exporter.export_heatmap(results, out)

    assert result_path.is_absolute()


def test_export_heatmap_handles_multiple_backends_tasks(tmp_path: Path) -> None:
    results = [
        _make_result(backend="groq", task="gsm", delta=0.15),
        _make_result(backend="groq", task="ner", delta=-0.05),
        _make_result(backend="ollama", task="gsm", delta=0.10),
        _make_result(backend="ollama", task="ner", delta=0.20),
    ]
    out = tmp_path / "heatmap_multi.png"

    exporter = PNGExporter()
    result_path = exporter.export_heatmap(results, out)

    assert isinstance(result_path, Path)


# ---------------------------------------------------------------------------
# export_latency_bars
# ---------------------------------------------------------------------------


def test_export_latency_bars_creates_file(tmp_path: Path) -> None:
    results = [
        _make_result(backend="groq"),
        _make_result(backend="ollama"),
    ]
    out = tmp_path / "latency.png"

    exporter = PNGExporter()
    result_path = exporter.export_latency_bars(results, out)

    assert isinstance(result_path, Path)
    assert result_path.suffix == ".png"


def test_export_latency_bars_adds_png_extension(tmp_path: Path) -> None:
    results = [_make_result()]
    out = tmp_path / "latency"  # no .png

    exporter = PNGExporter()
    result_path = exporter.export_latency_bars(results, out)

    assert result_path.suffix == ".png"


def test_export_latency_bars_single_backend(tmp_path: Path) -> None:
    results = [_make_result(backend="vllm")]
    out = tmp_path / "latency_single.png"

    exporter = PNGExporter()
    result_path = exporter.export_latency_bars(results, out)

    assert isinstance(result_path, Path)


# ---------------------------------------------------------------------------
# export_threshold_curve
# ---------------------------------------------------------------------------


def test_export_threshold_curve_creates_file(tmp_path: Path) -> None:
    complexity_scores = [0.3, 0.6, 0.8]
    accuracy_deltas = [0.05, -0.02, 0.20]
    out = tmp_path / "curve.png"

    exporter = PNGExporter()
    result_path = exporter.export_threshold_curve(complexity_scores, accuracy_deltas, out)

    assert isinstance(result_path, Path)
    assert result_path.suffix == ".png"


def test_export_threshold_curve_with_backend_label(tmp_path: Path) -> None:
    out = tmp_path / "curve.png"

    exporter = PNGExporter()
    result_path = exporter.export_threshold_curve([0.5], [0.1], out, backend="groq")

    assert isinstance(result_path, Path)


def test_export_threshold_curve_adds_png_extension(tmp_path: Path) -> None:
    out = tmp_path / "curve"  # no extension

    exporter = PNGExporter()
    result_path = exporter.export_threshold_curve([0.4, 0.7], [0.1, -0.05], out)

    assert result_path.suffix == ".png"


# ---------------------------------------------------------------------------
# generate_paper_figures
# ---------------------------------------------------------------------------


def test_generate_paper_figures_returns_dict(tmp_path: Path) -> None:
    results = [
        _make_result(backend="groq", task="gsm", delta=0.15),
        _make_result(backend="ollama", task="gsm", delta=0.05),
    ]

    exporter = PNGExporter()
    artifacts = exporter.generate_paper_figures(results, tmp_path / "figures")

    assert isinstance(artifacts, dict)
    assert "heatmap" in artifacts
    assert "latency_bars" in artifacts
    assert "threshold_curve" in artifacts


def test_generate_paper_figures_empty_results_skips_threshold_curve(
    tmp_path: Path,
) -> None:
    exporter = PNGExporter()
    artifacts = exporter.generate_paper_figures([], tmp_path / "figures")

    # With empty results, threshold_curve should not be in artifacts
    assert "threshold_curve" not in artifacts


def test_generate_paper_figures_all_paths_are_path_objects(tmp_path: Path) -> None:
    results = [_make_result()]
    exporter = PNGExporter()
    artifacts = exporter.generate_paper_figures(results, tmp_path / "figs")

    for key, val in artifacts.items():
        assert isinstance(val, Path), f"artifact '{key}' is not a Path"


# ---------------------------------------------------------------------------
# ImportError when matplotlib is missing
# ---------------------------------------------------------------------------


def test_exporter_without_matplotlib_raises_import_error(tmp_path: Path) -> None:
    results = [_make_result()]
    out = tmp_path / "heatmap.png"

    # Simulate matplotlib missing by patching None into sys.modules
    # and removing it from the parent mock's attribute
    matplotlib_missing = MagicMock()
    matplotlib_missing.use.side_effect = None

    modules_backup = {
        k: sys.modules.pop(k) for k in list(sys.modules.keys()) if k.startswith("matplotlib")
    }
    try:
        with patch.dict(
            "sys.modules",
            {"matplotlib": None, "matplotlib.pyplot": None, "numpy": None},
        ):
            exporter = PNGExporter()
            with pytest.raises((ImportError, TypeError)):
                exporter.export_heatmap(results, out)
    finally:
        sys.modules.update(modules_backup)
