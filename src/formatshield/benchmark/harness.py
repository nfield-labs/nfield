"""
BenchmarkHarness — runs FormatShield benchmark tasks across backends and
generates paper artifacts.

The harness drives the full benchmark loop:

1. For each (task, backend) pair, it calls the task's ``get_problems()``
   method to obtain a list of prompts with ground-truth answers.
2. For every problem it runs the model twice: once with Think-Then-Format
   (TTF) and once in direct-generation mode.
3. It scores both responses using the task's ``score_response()`` method.
4. It records a :class:`~formatshield.scorer.features.BenchmarkResult`
   capturing accuracy, latency, and routing metadata.
5. After all runs, it writes raw JSONL files, summary CSVs, and optional
   LaTeX table code for the paper.

The harness is backend-agnostic; pass any object that implements the
:class:`~formatshield.backends.protocol.Backend` protocol via the
``backend_obj`` / ``backend_objects`` parameters.  When no backend object is
supplied the :class:`~formatshield.backends.dryrun_backend.DryRunBackend` is
used automatically, making CI runs and unit tests possible without API keys.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from formatshield.scorer.features import BenchmarkResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic helpers (real logic — not simulation)
# ---------------------------------------------------------------------------


def _compute_complexity_score(task_name: str) -> float:
    """Return a deterministic complexity score for a task."""
    scores = {
        "gsm_symbolic": 0.82,
        "medical_ner": 0.68,
        "template_fill": 0.15,
    }
    return scores.get(task_name, 0.5)


def _detect_failure_modes(
    task_name: str,
    ttf_accuracy: float,
    direct_accuracy: float,
    overhead_pct: float,
    expected_ttf_benefit: bool,
) -> list[str]:
    """
    Heuristically detect failure modes from benchmark metrics.

    Returns a list of failure-mode label strings.
    """
    modes: list[str] = []
    accuracy_delta = ttf_accuracy - direct_accuracy

    # TTF expected to help but didn't
    if expected_ttf_benefit and accuracy_delta < -0.05:
        modes.append("ttf_accuracy_regression")

    # TTF not expected to help, but overhead was applied anyway
    if not expected_ttf_benefit and overhead_pct > 30.0:
        modes.append("unnecessary_ttf_overhead")

    # Large overhead with minimal accuracy gain
    if overhead_pct > 80.0 and accuracy_delta < 0.05:
        modes.append("high_overhead_low_gain")

    # TTF degraded accuracy on a simple task (routing error)
    if not expected_ttf_benefit and accuracy_delta < -0.03:
        modes.append("ttf_routing_error")

    return modes


def _get_problem_prompt(task: Any, problem: dict[str, Any]) -> str:
    """Extract the prompt string from a problem dict, using task.build_prompt() when available."""
    if hasattr(task, "build_prompt") and "question" in problem:
        return task.build_prompt(problem["question"])
    return str(problem.get("prompt", problem.get("question", "")))


def _get_task_schema(task: Any) -> dict[str, Any] | None:
    """Return a JSON Schema dict for *task*, or ``None`` if the task has no schema."""
    schema_attr = getattr(task, "schema", None)
    if schema_attr is None:
        return None
    # Pydantic BaseModel subclass
    if hasattr(schema_attr, "model_json_schema"):
        try:
            return schema_attr.model_json_schema()  # type: ignore[no-any-return]
        except Exception:
            logger.debug("_get_task_schema: model_json_schema() failed, skipping")
    # Plain dict schema
    if isinstance(schema_attr, dict):
        return schema_attr
    return None


def _parse_response(raw: str) -> dict[str, Any]:
    """Parse a JSON string to a dict, returning ``{}`` on failure."""
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


# ---------------------------------------------------------------------------
# BenchmarkHarness
# ---------------------------------------------------------------------------


class BenchmarkHarness:
    """
    Orchestrates the FormatShield benchmark across multiple tasks and backends.

    The harness is designed for reproducibility and uses real backend calls
    via the :class:`~formatshield.ttf.engine.TTFEngine` for TTF runs and
    direct ``backend.generate()`` calls for the direct-generation baseline.
    When no backend object is provided the
    :class:`~formatshield.backends.dryrun_backend.DryRunBackend` is used so
    that the full pipeline can be exercised without API keys.

    Parameters
    ----------
    output_dir:
        Root directory for all benchmark output files.  Created automatically
        if it does not exist.
    seed:
        Seed stored on the instance for reproducible DryRunBackend instances.

    Example::

        from formatshield.backends.dryrun_backend import DryRunBackend

        harness = BenchmarkHarness(output_dir=Path("benchmark_results"))
        results = asyncio.run(harness.run(
            tasks=["gsm_symbolic", "medical_ner", "template_fill"],
            backends=["dryrun"],
            models={"dryrun": "dryrun/default"},
            backend_objects={"dryrun": DryRunBackend(seed=42)},
            quick=True,
        ))
        artifacts = harness.generate_artifacts(results)
    """

    def __init__(
        self,
        output_dir: Path = Path("benchmark_results"),
        seed: int = 42,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "raw").mkdir(exist_ok=True)
        (self.output_dir / "artifacts").mkdir(exist_ok=True)
        self._seed = seed
        # Keep _rng for backwards compatibility with callers that inspect it
        self._rng = random.Random(seed)  # noqa: S311 — seeded for reproducibility, not cryptography
        logger.info("BenchmarkHarness initialised, output_dir=%s", self.output_dir)

    # ------------------------------------------------------------------
    # Core run methods
    # ------------------------------------------------------------------

    async def run_task_on_backend(
        self,
        task: Any,
        backend: str,
        model: str,
        quick: bool = False,
        backend_obj: Any | None = None,
    ) -> list[BenchmarkResult]:
        """
        Run a single task against a single backend and return per-problem results.

        For each problem returned by ``task.get_problems(quick)``:

        1. Runs FormatShield with TTF → scores → records TTF accuracy + latency.
        2. Runs FormatShield without TTF (direct) → scores → records direct accuracy + latency.
        3. Computes accuracy_delta and overhead_pct.
        4. Detects failure modes.
        5. Appends a :class:`BenchmarkResult` to the output list.

        Parameters
        ----------
        task:
            A task instance with ``name``, ``expected_ttf_benefit``,
            ``get_problems(quick)``, and ``score_response()`` attributes.
        backend:
            Backend identifier string (e.g. ``"groq"``).
        model:
            Model string passed to the backend (e.g. ``"groq/llama-3.1-70b-versatile"``).
        quick:
            When ``True``, uses the reduced problem set from ``task.get_problems()``.
        backend_obj:
            Optional backend instance implementing the Backend protocol.
            When ``None`` a :class:`~formatshield.backends.dryrun_backend.DryRunBackend`
            is used automatically.

        Returns
        -------
        list[BenchmarkResult]
            One result per problem in the task's problem set.
        """
        from formatshield.backends.dryrun_backend import DryRunBackend
        from formatshield.ttf.engine import TTFEngine

        effective_backend = (
            backend_obj if backend_obj is not None else DryRunBackend(seed=self._seed)
        )
        engine = TTFEngine(effective_backend)

        problems = task.get_problems(quick=quick)
        task_name: str = task.name
        expected_ttf_benefit: bool = task.expected_ttf_benefit
        complexity_score = _compute_complexity_score(task_name)
        schema_dict = _get_task_schema(task)

        results: list[BenchmarkResult] = []

        for i, problem in enumerate(problems):
            logger.debug(
                "run_task_on_backend: task=%s backend=%s problem=%d/%d",
                task_name,
                backend,
                i + 1,
                len(problems),
            )

            prompt = _get_problem_prompt(task, problem)
            ground_truth = (
                problem.get("answer") or problem.get("entities") or problem.get("expected")
            )

            # --- TTF run ---
            t0_ttf = time.monotonic()
            try:
                _, ttf_raw = await engine.generate(prompt, schema=schema_dict)
                ttf_response = _parse_response(ttf_raw)
            except Exception as exc:
                logger.warning(
                    "run_task_on_backend: TTF run failed (%s) — using empty response", exc
                )
                ttf_response = {}
            ttf_latency = (time.monotonic() - t0_ttf) * 1000.0

            # --- Direct run ---
            t0_direct = time.monotonic()
            try:
                direct_raw = await effective_backend.generate(
                    prompt, schema=schema_dict, constraints="json"
                )
                direct_response = _parse_response(direct_raw)
            except Exception as exc:
                logger.warning(
                    "run_task_on_backend: direct run failed (%s) — using empty response", exc
                )
                direct_response = {}
            direct_latency = (time.monotonic() - t0_direct) * 1000.0

            # Ensure latencies are positive (clock resolution guard)
            ttf_latency = max(ttf_latency, 0.001)
            direct_latency = max(direct_latency, 0.001)

            # --- Score both responses ---
            ttf_score = task.score_response(ttf_response, ground_truth)
            direct_score = task.score_response(direct_response, ground_truth)

            # --- Compute derived metrics ---
            accuracy_delta = ttf_score - direct_score
            overhead_pct = (ttf_latency - direct_latency) / direct_latency * 100.0

            # --- Failure mode detection ---
            failure_modes = _detect_failure_modes(
                task_name=task_name,
                ttf_accuracy=ttf_score,
                direct_accuracy=direct_score,
                overhead_pct=overhead_pct,
                expected_ttf_benefit=expected_ttf_benefit,
            )

            result = BenchmarkResult(
                task=task_name,
                backend=backend,
                model=model,
                direct_accuracy=direct_score,
                ttf_accuracy=ttf_score,
                accuracy_delta=accuracy_delta,
                direct_latency_ms=round(direct_latency, 2),
                ttf_latency_ms=round(ttf_latency, 2),
                overhead_pct=round(overhead_pct, 2),
                complexity_score=complexity_score,
                failure_modes_detected=failure_modes,
            )
            results.append(result)

            # Brief cooperative yield to avoid blocking the event loop
            await asyncio.sleep(0)

        return results

    async def run(
        self,
        tasks: list[str],
        backends: list[str],
        models: dict[str, str],
        quick: bool = False,
        backend_objects: dict[str, Any] | None = None,
    ) -> list[BenchmarkResult]:
        """
        Run all task × backend combinations and return aggregated results.

        Uses :func:`asyncio.gather` for concurrent execution of backend calls.
        All results are written to JSONL files in ``output_dir/raw/`` and a
        summary CSV is written to ``output_dir/summary.csv``.

        Parameters
        ----------
        tasks:
            List of task names to run.  Must match one of: ``"gsm_symbolic"``,
            ``"medical_ner"``, ``"template_fill"``.
        backends:
            List of backend names to run (e.g. ``["groq", "ollama"]``).
        models:
            Dict mapping backend name → model identifier string.
        quick:
            When ``True``, each task uses its reduced problem set for fast runs.
        backend_objects:
            Optional mapping of backend name → backend instance.  When a
            backend name is not present in this dict, a
            :class:`~formatshield.backends.dryrun_backend.DryRunBackend` is
            used for that backend automatically.

        Returns
        -------
        list[BenchmarkResult]
            All individual problem-level results across all tasks and backends.
        """
        # Lazy import to avoid circular dependency issues at module load time
        from formatshield.benchmark.exporters import CSVExporter
        from formatshield.benchmark.tasks import (
            GSMSymbolicTask,
            MedicalNERTask,
            TemplateFillTask,
        )

        _task_registry: dict[str, Any] = {
            "gsm_symbolic": GSMSymbolicTask(),
            "gsm": GSMSymbolicTask(),  # alias
            "medical_ner": MedicalNERTask(),
            "template_fill": TemplateFillTask(),
        }

        task_objects = []
        for name in tasks:
            if name not in _task_registry:
                logger.warning("Unknown task %r — skipping", name)
                continue
            task_objects.append(_task_registry[name])

        effective_backend_objects = backend_objects or {}

        all_results: list[BenchmarkResult] = []
        coroutines = []

        for task_obj in task_objects:
            for backend in backends:
                model = models.get(backend, f"{backend}/default")
                backend_obj = effective_backend_objects.get(backend)
                coroutines.append(
                    self.run_task_on_backend(
                        task_obj, backend, model, quick=quick, backend_obj=backend_obj
                    )
                )

        # Run all (task, backend) pairs concurrently
        batch_results = await asyncio.gather(*coroutines, return_exceptions=True)

        for item in batch_results:
            if isinstance(item, Exception):
                logger.error("run: coroutine raised exception: %s", item)
            elif isinstance(item, list):
                all_results.extend(item)

        # Write raw JSONL
        self._write_raw_jsonl(all_results)

        # Write summary CSV
        exporter = CSVExporter()
        summary_path = exporter.export_summary(
            all_results,
            self.output_dir / "summary.csv",
        )
        logger.info("run: summary CSV written to %s", summary_path)

        return all_results

    def generate_artifacts(
        self,
        results: list[BenchmarkResult],
    ) -> dict[str, Path]:
        """
        Generate paper-ready artifact files from a completed set of results.

        Writes the following files under ``output_dir/artifacts/``:

        * ``table1_accuracy_by_backend.csv`` — backend × task accuracy table.
        * ``table2_failure_modes.csv`` — rows where failure modes were detected.
        * ``summary.json`` — machine-readable JSON summary.
        * ``table1_latex.tex`` — LaTeX table code for the paper.

        Parameters
        ----------
        results:
            Full result list returned by :meth:`run`.

        Returns
        -------
        dict[str, Path]
            Mapping of artifact name → absolute file path.
        """
        from formatshield.benchmark.exporters import CSVExporter

        exporter = CSVExporter()
        artifacts_dir = self.output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        artifacts: dict[str, Path] = {}

        # Table 1: accuracy by backend
        table1_path = exporter.export_summary(
            results,
            artifacts_dir / "table1_accuracy_by_backend.csv",
        )
        artifacts["table1_accuracy_by_backend"] = table1_path

        # Table 2: failure modes
        table2_path = exporter.export_failure_modes(
            results,
            artifacts_dir / "table2_failure_modes.csv",
        )
        artifacts["table2_failure_modes"] = table2_path

        # Summary JSON
        summary_path = exporter.generate_summary_json(
            results,
            artifacts_dir / "summary.json",
        )
        artifacts["summary_json"] = summary_path

        # LaTeX table
        latex_code = exporter.generate_latex_table(results)
        latex_path = artifacts_dir / "table1_latex.tex"
        latex_path.write_text(latex_code, encoding="utf-8")
        artifacts["table1_latex"] = latex_path

        logger.info(
            "generate_artifacts: wrote %d artifact files to %s",
            len(artifacts),
            artifacts_dir,
        )
        return artifacts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_raw_jsonl(self, results: list[BenchmarkResult]) -> Path:
        """
        Write all results to a JSONL file under ``output_dir/raw/``.

        Returns the path to the written file.
        """
        import datetime

        timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = self.output_dir / "raw" / f"results_{timestamp}.jsonl"

        with out_path.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r.to_dict()) + "\n")

        logger.info("_write_raw_jsonl: wrote %d rows to %s", len(results), out_path)
        return out_path
