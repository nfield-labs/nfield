"""
Unit tests for the judge-wiring added to BenchmarkHarness.

Verifies:
- _JUDGE_TASKS frozenset contains the right task names
- BenchmarkHarness stores the judge on self._judge
- Without a judge, harness falls back to task.score_response()
- With a judge, harness calls judge.ajudge() for tasks in _JUDGE_TASKS
- With a judge, non-judge tasks still use task.score_response()
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from formatshield.benchmark.harness import _JUDGE_TASKS, BenchmarkHarness

# ---------------------------------------------------------------------------
# _JUDGE_TASKS constant
# ---------------------------------------------------------------------------


def test_judge_tasks_is_frozenset() -> None:
    assert isinstance(_JUDGE_TASKS, frozenset)


def test_judge_tasks_contains_gsm_symbolic() -> None:
    assert "gsm_symbolic" in _JUDGE_TASKS


def test_judge_tasks_contains_math500() -> None:
    assert "math500" in _JUDGE_TASKS


def test_judge_tasks_contains_legal_extract() -> None:
    assert "legal_extract" in _JUDGE_TASKS


def test_judge_tasks_contains_sql_extraction() -> None:
    assert "sql_extraction" in _JUDGE_TASKS


def test_judge_tasks_contains_zebralogic() -> None:
    assert "zebralogic" in _JUDGE_TASKS


def test_judge_tasks_does_not_contain_template_fill() -> None:
    """template_fill is a simple task — rule-based scoring is fine."""
    assert "template_fill" not in _JUDGE_TASKS


def test_judge_tasks_does_not_contain_medical_ner() -> None:
    assert "medical_ner" not in _JUDGE_TASKS


# ---------------------------------------------------------------------------
# BenchmarkHarness construction
# ---------------------------------------------------------------------------


def test_harness_stores_judge() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_judge = MagicMock()
        harness = BenchmarkHarness(output_dir=Path(tmpdir), judge=mock_judge)
        assert harness._judge is mock_judge


def test_harness_judge_defaults_to_none() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        harness = BenchmarkHarness(output_dir=Path(tmpdir))
        assert harness._judge is None


# ---------------------------------------------------------------------------
# Helpers — minimal task and backend mocks
# ---------------------------------------------------------------------------


def _make_task(name: str, expected_ttf_benefit: bool = True, score: float = 0.5) -> Any:
    task = MagicMock()
    task.name = name
    task.expected_ttf_benefit = expected_ttf_benefit
    task.schema = None
    task.get_problems = MagicMock(
        return_value=[{"question": "test question?", "answer": "42"}]
    )
    task.build_prompt = MagicMock(return_value="test question?")
    task.score_response = MagicMock(return_value=score)
    return task


def _make_backend(response: str = '{"answer": 42}') -> Any:
    backend = MagicMock()
    backend.name = "mock"
    backend.generate = AsyncMock(return_value=response)
    backend.stream = AsyncMock()
    return backend


def _make_engine_patch(thinking: str = "thinking", response: str = '{"answer": 42}') -> Any:
    engine = MagicMock()
    engine.generate = AsyncMock(return_value=(thinking, response))
    return engine


# ---------------------------------------------------------------------------
# Without judge — uses task.score_response()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_harness_no_judge_uses_task_score_response() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        harness = BenchmarkHarness(output_dir=Path(tmpdir))
        task = _make_task("gsm_symbolic", score=0.75)
        backend = _make_backend()

        with patch("formatshield.ttf.engine.TTFEngine") as mock_engine:
            mock_engine.return_value = _make_engine_patch()
            results = await harness.run_task_on_backend(
                task, "mock", "mock/model", quick=False, backend_obj=backend
            )

    assert len(results) == 1
    task.score_response.assert_called()
    assert results[0].ttf_accuracy == 0.75


# ---------------------------------------------------------------------------
# With judge — uses ajudge() for judge tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_harness_judge_called_for_gsm_symbolic() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_judge = MagicMock()
        mock_judge.ajudge = AsyncMock(return_value=True)

        harness = BenchmarkHarness(output_dir=Path(tmpdir), judge=mock_judge)
        task = _make_task("gsm_symbolic", score=0.5)
        backend = _make_backend()

        with patch("formatshield.ttf.engine.TTFEngine") as mock_engine:
            mock_engine.return_value = _make_engine_patch()
            results = await harness.run_task_on_backend(
                task, "mock", "mock/model", quick=False, backend_obj=backend
            )

    assert mock_judge.ajudge.called
    # Both TTF and direct judged CORRECT → scores should be 1.0
    assert results[0].ttf_accuracy == 1.0
    assert results[0].direct_accuracy == 1.0


@pytest.mark.asyncio
async def test_harness_judge_incorrect_verdict() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_judge = MagicMock()
        mock_judge.ajudge = AsyncMock(return_value=False)

        harness = BenchmarkHarness(output_dir=Path(tmpdir), judge=mock_judge)
        task = _make_task("math500")
        backend = _make_backend()

        with patch("formatshield.ttf.engine.TTFEngine") as mock_engine:
            mock_engine.return_value = _make_engine_patch()
            results = await harness.run_task_on_backend(
                task, "mock", "mock/model", quick=False, backend_obj=backend
            )

    assert results[0].ttf_accuracy == 0.0
    assert results[0].direct_accuracy == 0.0


@pytest.mark.asyncio
async def test_harness_judge_not_called_for_template_fill() -> None:
    """template_fill is not in _JUDGE_TASKS — rule-based scoring used."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_judge = MagicMock()
        mock_judge.ajudge = AsyncMock(return_value=True)

        harness = BenchmarkHarness(output_dir=Path(tmpdir), judge=mock_judge)
        task = _make_task("template_fill", expected_ttf_benefit=False, score=0.9)
        backend = _make_backend()

        with patch("formatshield.ttf.engine.TTFEngine") as mock_engine:
            mock_engine.return_value = _make_engine_patch()
            await harness.run_task_on_backend(
                task, "mock", "mock/model", quick=False, backend_obj=backend
            )

    # judge.ajudge must not have been called for a non-judge task
    mock_judge.ajudge.assert_not_called()
    # task.score_response should have been called instead
    task.score_response.assert_called()


@pytest.mark.asyncio
async def test_harness_judge_called_twice_per_problem() -> None:
    """ajudge is called once for TTF response and once for direct response."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_judge = MagicMock()
        mock_judge.ajudge = AsyncMock(return_value=True)

        harness = BenchmarkHarness(output_dir=Path(tmpdir), judge=mock_judge)
        task = _make_task("legal_extract")
        backend = _make_backend()

        with patch("formatshield.ttf.engine.TTFEngine") as mock_engine:
            mock_engine.return_value = _make_engine_patch()
            await harness.run_task_on_backend(
                task, "mock", "mock/model", quick=False, backend_obj=backend
            )

    assert mock_judge.ajudge.call_count == 2


@pytest.mark.asyncio
async def test_harness_accuracy_delta_computed_from_judge_scores() -> None:
    """accuracy_delta should be judge_ttf - judge_direct."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # TTF → CORRECT (1.0), direct → INCORRECT (0.0)
        mock_judge = MagicMock()
        mock_judge.ajudge = AsyncMock(side_effect=[True, False])

        harness = BenchmarkHarness(output_dir=Path(tmpdir), judge=mock_judge)
        task = _make_task("gsm_symbolic")
        backend = _make_backend()

        with patch("formatshield.ttf.engine.TTFEngine") as mock_engine:
            mock_engine.return_value = _make_engine_patch()
            results = await harness.run_task_on_backend(
                task, "mock", "mock/model", quick=False, backend_obj=backend
            )

    assert results[0].ttf_accuracy == 1.0
    assert results[0].direct_accuracy == 0.0
    assert results[0].accuracy_delta == pytest.approx(1.0)
