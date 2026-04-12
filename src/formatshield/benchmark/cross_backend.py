"""CrossBackendBenchmark — measures format tax across all supported backends."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from formatshield.scorer.features import BenchmarkResult

logger = logging.getLogger(__name__)

_DEFAULT_MODELS: dict[str, str] = {
    "groq": "groq/llama-3.1-70b-versatile",
    "ollama": "ollama/llama3.1:70b",
    "openrouter": "openrouter/meta-llama/llama-3.1-70b-instruct",
    "vllm": "vllm/meta-llama/Llama-3-70b-Instruct",
}


class CrossBackendBenchmark:
    """Runs a benchmark task across multiple backends and compares format tax.

    Example::

        bench = CrossBackendBenchmark(backends=["groq", "ollama"])
        results = await bench.run(task="gsm", quick=True)
        tax_table = bench.get_format_tax_table(results)
    """

    def __init__(
        self,
        backends: list[str],
        models: dict[str, str] | None = None,
    ) -> None:
        self.backends = backends
        self.models = models or {b: _DEFAULT_MODELS.get(b, f"{b}/default") for b in backends}

    async def run(
        self,
        task: str = "gsm",
        quick: bool = False,
    ) -> list[BenchmarkResult]:
        """Run *task* across all configured backends concurrently."""
        from pathlib import Path

        from formatshield.benchmark.harness import BenchmarkHarness

        harness = BenchmarkHarness(output_dir=Path("benchmark_results"))

        async def _run_one(backend: str) -> list[BenchmarkResult]:
            try:
                return await harness.run_task_on_backend(
                    task=task,
                    backend=backend,
                    model=self.models[backend],
                    quick=quick,
                )
            except Exception as exc:
                logger.warning("CrossBackendBenchmark: backend %s failed: %s", backend, exc)
                return []

        tasks = [_run_one(b) for b in self.backends]
        nested = await asyncio.gather(*tasks)
        return [r for results in nested for r in results]

    def get_format_tax_table(self, results: list[BenchmarkResult]) -> dict[str, float]:
        """Aggregate accuracy_delta by backend.

        Returns a dict mapping backend name to mean accuracy delta
        (positive = TTF helps, negative = TTF hurts).
        """
        from collections import defaultdict

        deltas: dict[str, list[float]] = defaultdict(list)
        for r in results:
            deltas[r.backend].append(r.accuracy_delta)

        return {backend: sum(vals) / len(vals) for backend, vals in deltas.items() if vals}

    async def train_oracle(self, results: list[BenchmarkResult]) -> Any:
        """Train ThresholdOracle from benchmark results and save to disk.

        Returns the trained oracle instance.
        """
        import csv
        import tempfile
        from pathlib import Path

        from formatshield.oracle.threshold_oracle import ThresholdOracle

        # Write results to a temp CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            tmp_path = Path(f.name)
            if results:
                writer = csv.DictWriter(f, fieldnames=list(results[0].to_dict().keys()))
                writer.writeheader()
                for r in results:
                    writer.writerow(r.to_dict())

        oracle = ThresholdOracle.from_benchmark_data(tmp_path)
        tmp_path.unlink(missing_ok=True)
        return oracle
