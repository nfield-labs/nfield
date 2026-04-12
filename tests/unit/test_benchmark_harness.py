"""
Unit tests for formatshield.benchmark.harness.

Covers BenchmarkHarness initialisation, per-backend task running,
full run(), artifact generation, and all internal simulation helpers.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from formatshield.benchmark.harness import (
    BenchmarkHarness,
    _compute_complexity_score,
    _detect_failure_modes,
    _simulate_direct_response,
    _simulate_ttf_response,
)
from formatshield.scorer.features import BenchmarkResult

# ---------------------------------------------------------------------------
# Minimal mock task (no external imports required)
# ---------------------------------------------------------------------------


class _MockTask:
    name = "gsm_symbolic"
    expected_ttf_benefit = True

    def get_problems(self, quick: bool = False) -> list[dict[str, Any]]:
        return [{"prompt": "What is 2+2?", "answer": 4.0}] * (3 if quick else 5)

    def score_response(self, response: Any, ground_truth: Any) -> float:
        if isinstance(response, dict) and "final_answer" in response:
            return 1.0 if abs(response["final_answer"] - float(ground_truth)) < 0.01 else 0.0
        return 0.0


class _MockSimpleTask:
    """A generic, non-reasoning task (expected_ttf_benefit=False)."""

    name = "template_fill"
    expected_ttf_benefit = False

    def get_problems(self, quick: bool = False) -> list[dict[str, Any]]:
        return [{"prompt": "Fill in the blank.", "answer": "hello"}] * 2

    def score_response(self, response: Any, ground_truth: Any) -> float:
        return 1.0 if response == {"answer": ground_truth} else 0.0


# ---------------------------------------------------------------------------
# Helper: build a minimal BenchmarkResult
# ---------------------------------------------------------------------------


def _make_result(
    *,
    task: str = "gsm_symbolic",
    backend: str = "groq",
    model: str = "groq/llama3",
    direct_accuracy: float = 0.6,
    ttf_accuracy: float = 0.8,
    failure_modes: list[str] | None = None,
) -> BenchmarkResult:
    delta = ttf_accuracy - direct_accuracy
    return BenchmarkResult(
        task=task,
        backend=backend,
        model=model,
        direct_accuracy=direct_accuracy,
        ttf_accuracy=ttf_accuracy,
        accuracy_delta=delta,
        direct_latency_ms=200.0,
        ttf_latency_ms=500.0,
        overhead_pct=150.0,
        complexity_score=0.82,
        failure_modes_detected=failure_modes or [],
    )


# ---------------------------------------------------------------------------
# BenchmarkHarness.__init__
# ---------------------------------------------------------------------------


def test_harness_creates_output_dir(tmp_path: Path) -> None:
    """BenchmarkHarness.__init__ must create the output directory."""
    out = tmp_path / "bench_out"
    assert not out.exists()
    BenchmarkHarness(output_dir=out)
    assert out.is_dir()


def test_harness_creates_raw_subdir(tmp_path: Path) -> None:
    """BenchmarkHarness.__init__ must create a 'raw' subdirectory."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    assert (harness.output_dir / "raw").is_dir()


def test_harness_creates_artifacts_subdir(tmp_path: Path) -> None:
    """BenchmarkHarness.__init__ must create an 'artifacts' subdirectory."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    assert (harness.output_dir / "artifacts").is_dir()


def test_harness_seed_is_stored(tmp_path: Path) -> None:
    """The harness must store an RNG seeded with the given seed."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out", seed=99)
    # The RNG must be a random.Random instance (not the module-level Random)
    assert isinstance(harness._rng, random.Random)


# ---------------------------------------------------------------------------
# run_task_on_backend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_task_on_backend_returns_list(tmp_path: Path) -> None:
    """run_task_on_backend must return a non-empty list."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    results = await harness.run_task_on_backend(
        task=_MockTask(), backend="groq", model="groq/llama3", quick=True
    )
    assert isinstance(results, list)
    assert len(results) > 0


@pytest.mark.asyncio
async def test_run_task_on_backend_result_type(tmp_path: Path) -> None:
    """Every item returned by run_task_on_backend must be a BenchmarkResult."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    results = await harness.run_task_on_backend(
        task=_MockTask(), backend="groq", model="groq/llama3", quick=True
    )
    for r in results:
        assert isinstance(r, BenchmarkResult)


@pytest.mark.asyncio
async def test_run_task_on_backend_quick_problem_count(tmp_path: Path) -> None:
    """quick=True uses the reduced problem set (3 problems for _MockTask)."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    results = await harness.run_task_on_backend(
        task=_MockTask(), backend="groq", model="groq/llama3", quick=True
    )
    assert len(results) == 3


@pytest.mark.asyncio
async def test_run_task_on_backend_full_problem_count(tmp_path: Path) -> None:
    """quick=False uses the full problem set (5 problems for _MockTask)."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    results = await harness.run_task_on_backend(
        task=_MockTask(), backend="groq", model="groq/llama3", quick=False
    )
    assert len(results) == 5


@pytest.mark.asyncio
async def test_run_task_on_backend_result_fields(tmp_path: Path) -> None:
    """BenchmarkResult objects must carry the correct backend / task / model."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    results = await harness.run_task_on_backend(
        task=_MockTask(), backend="groq", model="groq/llama3", quick=True
    )
    for r in results:
        assert r.backend == "groq"
        assert r.task == "gsm_symbolic"
        assert r.model == "groq/llama3"


@pytest.mark.asyncio
async def test_run_task_on_backend_accuracy_delta_consistent(tmp_path: Path) -> None:
    """accuracy_delta must equal ttf_accuracy minus direct_accuracy."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    results = await harness.run_task_on_backend(
        task=_MockTask(), backend="groq", model="groq/llama3", quick=True
    )
    for r in results:
        assert abs(r.accuracy_delta - (r.ttf_accuracy - r.direct_accuracy)) < 1e-9


@pytest.mark.asyncio
async def test_run_task_on_backend_latencies_positive(tmp_path: Path) -> None:
    """Both direct_latency_ms and ttf_latency_ms must be positive."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    results = await harness.run_task_on_backend(
        task=_MockTask(), backend="groq", model="groq/llama3", quick=True
    )
    for r in results:
        assert r.direct_latency_ms > 0.0
        assert r.ttf_latency_ms > 0.0


@pytest.mark.asyncio
async def test_run_task_on_backend_failure_modes_is_list(tmp_path: Path) -> None:
    """failure_modes_detected must always be a list."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    results = await harness.run_task_on_backend(
        task=_MockTask(), backend="groq", model="groq/llama3", quick=True
    )
    for r in results:
        assert isinstance(r.failure_modes_detected, list)


@pytest.mark.asyncio
async def test_run_task_on_backend_complexity_score(tmp_path: Path) -> None:
    """complexity_score for gsm_symbolic must be 0.82 (deterministic)."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    results = await harness.run_task_on_backend(
        task=_MockTask(), backend="groq", model="groq/llama3", quick=True
    )
    for r in results:
        assert r.complexity_score == pytest.approx(0.82)


# ---------------------------------------------------------------------------
# BenchmarkHarness.run() (uses mock task objects passed via run_task_on_backend)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_all_results_for_known_tasks(tmp_path: Path) -> None:
    """run() with valid task names returns a non-empty result list."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    results = await harness.run(
        tasks=["gsm_symbolic"],
        backends=["groq"],
        models={"groq": "groq/llama3"},
        quick=True,
    )
    assert isinstance(results, list)
    assert len(results) > 0


@pytest.mark.asyncio
async def test_run_skips_unknown_tasks(tmp_path: Path) -> None:
    """run() with an unknown task name must return an empty list."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    results = await harness.run(
        tasks=["nonexistent_task_xyz"],
        backends=["groq"],
        models={"groq": "groq/llama3"},
        quick=True,
    )
    assert results == []


@pytest.mark.asyncio
async def test_run_multiple_backends_produces_results_for_each(tmp_path: Path) -> None:
    """run() with two backends should produce results for both."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    results = await harness.run(
        tasks=["gsm_symbolic"],
        backends=["groq", "ollama"],
        models={"groq": "groq/llama3", "ollama": "ollama/llama3"},
        quick=True,
    )
    backends_seen = {r.backend for r in results}
    assert "groq" in backends_seen
    assert "ollama" in backends_seen


@pytest.mark.asyncio
async def test_run_writes_summary_csv(tmp_path: Path) -> None:
    """run() must write a summary.csv to the output directory."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    await harness.run(
        tasks=["gsm_symbolic"],
        backends=["groq"],
        models={"groq": "groq/llama3"},
        quick=True,
    )
    assert (harness.output_dir / "summary.csv").exists()


@pytest.mark.asyncio
async def test_run_writes_raw_jsonl(tmp_path: Path) -> None:
    """run() must write a JSONL file under raw/."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    await harness.run(
        tasks=["gsm_symbolic"],
        backends=["groq"],
        models={"groq": "groq/llama3"},
        quick=True,
    )
    raw_files = list((harness.output_dir / "raw").glob("*.jsonl"))
    assert len(raw_files) >= 1


# ---------------------------------------------------------------------------
# BenchmarkHarness.generate_artifacts()
# ---------------------------------------------------------------------------


def test_generate_artifacts_returns_dict(tmp_path: Path) -> None:
    """generate_artifacts() must return a dict mapping artifact names to paths."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    results = [_make_result()]
    artifacts = harness.generate_artifacts(results)
    assert isinstance(artifacts, dict)
    assert len(artifacts) > 0


def test_generate_artifacts_table1_csv_exists(tmp_path: Path) -> None:
    """generate_artifacts() must produce table1_accuracy_by_backend.csv."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    artifacts = harness.generate_artifacts([_make_result()])
    assert artifacts["table1_accuracy_by_backend"].exists()


def test_generate_artifacts_failure_modes_csv_exists(tmp_path: Path) -> None:
    """generate_artifacts() must produce table2_failure_modes.csv."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    artifacts = harness.generate_artifacts([_make_result()])
    assert artifacts["table2_failure_modes"].exists()


def test_generate_artifacts_summary_json_exists(tmp_path: Path) -> None:
    """generate_artifacts() must produce summary.json."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    artifacts = harness.generate_artifacts([_make_result()])
    assert artifacts["summary_json"].exists()


def test_generate_artifacts_latex_tex_exists(tmp_path: Path) -> None:
    """generate_artifacts() must produce a .tex file."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    artifacts = harness.generate_artifacts([_make_result()])
    assert artifacts["table1_latex"].exists()


def test_generate_artifacts_empty_results(tmp_path: Path) -> None:
    """generate_artifacts() must not raise when results list is empty."""
    harness = BenchmarkHarness(output_dir=tmp_path / "out")
    artifacts = harness.generate_artifacts([])
    assert isinstance(artifacts, dict)


# ---------------------------------------------------------------------------
# _simulate_ttf_response
# ---------------------------------------------------------------------------


def test_simulate_ttf_response_returns_tuple() -> None:
    """_simulate_ttf_response must return a (dict, float) tuple."""
    rng = random.Random(0)
    problem = {"prompt": "Solve x+1=3", "answer": 2.0}
    result = _simulate_ttf_response(problem, "gsm_symbolic", "groq", rng)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_simulate_ttf_response_dict_type() -> None:
    """The response component must be a dict."""
    rng = random.Random(1)
    problem = {"prompt": "What is 3*3?", "answer": 9.0}
    response, _ = _simulate_ttf_response(problem, "gsm_symbolic", "groq", rng)
    assert isinstance(response, dict)


def test_simulate_ttf_response_latency_positive() -> None:
    """TTF latency must be positive."""
    rng = random.Random(2)
    problem = {"prompt": "Compute 5+5", "answer": 10.0}
    _, latency = _simulate_ttf_response(problem, "gsm_symbolic", "groq", rng)
    assert latency > 0.0


def test_simulate_ttf_response_gsm_symbolic_has_final_answer() -> None:
    """gsm_symbolic TTF response must contain 'final_answer' key."""
    rng = random.Random(3)
    problem = {"prompt": "2+2?", "answer": 4.0}
    response, _ = _simulate_ttf_response(problem, "gsm_symbolic", "groq", rng)
    assert "final_answer" in response


def test_simulate_ttf_response_higher_latency_than_direct() -> None:
    """TTF latency must exceed direct latency on the same RNG sequence."""
    rng_ttf = random.Random(42)
    rng_direct = random.Random(42)
    problem = {"prompt": "What is 1+1?", "answer": 2.0}
    _, ttf_lat = _simulate_ttf_response(problem, "gsm_symbolic", "groq", rng_ttf)
    _, direct_lat = _simulate_direct_response(problem, "gsm_symbolic", "groq", rng_direct)
    # TTF has 200-800 ms overhead; direct has 10-80 ms overhead — TTF always higher
    assert ttf_lat > direct_lat


# ---------------------------------------------------------------------------
# _simulate_direct_response
# ---------------------------------------------------------------------------


def test_simulate_direct_response_returns_tuple() -> None:
    """_simulate_direct_response must return a (dict, float) tuple."""
    rng = random.Random(10)
    problem = {"prompt": "What is 7-3?", "answer": 4.0}
    result = _simulate_direct_response(problem, "gsm_symbolic", "groq", rng)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_simulate_direct_response_dict_type() -> None:
    """The response component must be a dict."""
    rng = random.Random(11)
    problem = {"prompt": "8/2?", "answer": 4.0}
    response, _ = _simulate_direct_response(problem, "gsm_symbolic", "groq", rng)
    assert isinstance(response, dict)


def test_simulate_direct_response_latency_positive() -> None:
    """Direct latency must be positive."""
    rng = random.Random(12)
    problem = {"prompt": "Compute 6+6", "answer": 12.0}
    _, latency = _simulate_direct_response(problem, "gsm_symbolic", "groq", rng)
    assert latency > 0.0


def test_simulate_direct_response_generic_task() -> None:
    """_simulate_direct_response handles a generic (unknown) task name."""
    rng = random.Random(13)
    problem = {"answer": "unknown_answer"}
    response, latency = _simulate_direct_response(problem, "unknown_task", "groq", rng)
    assert isinstance(response, dict)
    assert latency > 0.0


# ---------------------------------------------------------------------------
# _detect_failure_modes
# ---------------------------------------------------------------------------


def test_detect_failure_modes_empty_when_no_issues() -> None:
    """No failure modes should be detected when everything is normal."""
    modes = _detect_failure_modes(
        task_name="gsm_symbolic",
        ttf_accuracy=0.85,
        direct_accuracy=0.70,
        overhead_pct=40.0,
        expected_ttf_benefit=True,
    )
    assert modes == []


def test_detect_failure_modes_ttf_regression() -> None:
    """ttf_accuracy_regression detected when TTF expected to help but didn't."""
    modes = _detect_failure_modes(
        task_name="gsm_symbolic",
        ttf_accuracy=0.50,
        direct_accuracy=0.70,
        overhead_pct=40.0,
        expected_ttf_benefit=True,
    )
    assert "ttf_accuracy_regression" in modes


def test_detect_failure_modes_unnecessary_overhead() -> None:
    """unnecessary_ttf_overhead detected when TTF not expected but overhead >30%."""
    modes = _detect_failure_modes(
        task_name="template_fill",
        ttf_accuracy=0.80,
        direct_accuracy=0.80,
        overhead_pct=50.0,
        expected_ttf_benefit=False,
    )
    assert "unnecessary_ttf_overhead" in modes


def test_detect_failure_modes_high_overhead_low_gain() -> None:
    """high_overhead_low_gain detected when overhead >80% and accuracy gain <5%."""
    modes = _detect_failure_modes(
        task_name="gsm_symbolic",
        ttf_accuracy=0.71,
        direct_accuracy=0.70,
        overhead_pct=90.0,
        expected_ttf_benefit=True,
    )
    assert "high_overhead_low_gain" in modes


def test_detect_failure_modes_routing_error() -> None:
    """ttf_routing_error detected when TTF not expected and hurts accuracy."""
    modes = _detect_failure_modes(
        task_name="template_fill",
        ttf_accuracy=0.60,
        direct_accuracy=0.70,
        overhead_pct=10.0,
        expected_ttf_benefit=False,
    )
    assert "ttf_routing_error" in modes


def test_detect_failure_modes_returns_list() -> None:
    """_detect_failure_modes must always return a list."""
    modes = _detect_failure_modes(
        task_name="gsm_symbolic",
        ttf_accuracy=0.9,
        direct_accuracy=0.5,
        overhead_pct=30.0,
        expected_ttf_benefit=True,
    )
    assert isinstance(modes, list)


def test_detect_failure_modes_multiple_can_coexist() -> None:
    """Multiple failure modes can be detected simultaneously."""
    modes = _detect_failure_modes(
        task_name="template_fill",
        ttf_accuracy=0.50,
        direct_accuracy=0.70,
        overhead_pct=90.0,
        expected_ttf_benefit=False,
    )
    # unnecessary_ttf_overhead + high_overhead_low_gain + ttf_routing_error
    assert len(modes) >= 2


# ---------------------------------------------------------------------------
# _compute_complexity_score
# ---------------------------------------------------------------------------


def test_compute_complexity_score_gsm_symbolic() -> None:
    """gsm_symbolic must return 0.82."""
    assert _compute_complexity_score("gsm_symbolic") == pytest.approx(0.82)


def test_compute_complexity_score_medical_ner() -> None:
    """medical_ner must return 0.68."""
    assert _compute_complexity_score("medical_ner") == pytest.approx(0.68)


def test_compute_complexity_score_template_fill() -> None:
    """template_fill must return 0.15."""
    assert _compute_complexity_score("template_fill") == pytest.approx(0.15)


def test_compute_complexity_score_unknown_task() -> None:
    """Unknown tasks must return the default score of 0.5."""
    assert _compute_complexity_score("unknown_task_xyz") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# BenchmarkResult.accuracy_delta correctness
# ---------------------------------------------------------------------------


def test_benchmark_result_accuracy_delta_positive() -> None:
    """accuracy_delta should be positive when TTF outperforms direct."""
    r = _make_result(direct_accuracy=0.6, ttf_accuracy=0.8)
    assert r.accuracy_delta == pytest.approx(0.2)


def test_benchmark_result_accuracy_delta_negative() -> None:
    """accuracy_delta should be negative when direct outperforms TTF."""
    r = _make_result(direct_accuracy=0.9, ttf_accuracy=0.7)
    assert r.accuracy_delta == pytest.approx(-0.2)


def test_benchmark_result_accuracy_delta_zero() -> None:
    """accuracy_delta should be zero when both are equal."""
    r = _make_result(direct_accuracy=0.75, ttf_accuracy=0.75)
    assert r.accuracy_delta == pytest.approx(0.0)


def test_benchmark_result_to_dict_has_all_fields() -> None:
    """to_dict() must include all required keys."""
    r = _make_result()
    d = r.to_dict()
    expected_keys = {
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
    assert expected_keys.issubset(d.keys())
