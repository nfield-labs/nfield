"""
PNGExporter — generates matplotlib visualisations from FormatShield benchmark results.

The primary output is a heatmap (backends x tasks x accuracy_delta) — the image
that goes into every launch post and into the paper as Figure 1.

Requires the ``benchmark`` optional extra:
    pip install formatshield[benchmark]
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from formatshield.scorer.features import BenchmarkResult

logger = logging.getLogger(__name__)

# Green = TTF helps. Red = TTF hurts. White = no data.
_CMAP = "RdYlGn"

# Colorbar limits — ±30pp accuracy delta covers all known empirical ranges.
_VMIN: float = -0.30
_VMAX: float = 0.30


class PNGExporter:
    """
    Generates PNG visualisations from :class:`~formatshield.scorer.features.BenchmarkResult`
    lists.

    All methods are stateless; the class exists purely for namespace organisation.

    Requires ``matplotlib`` and ``numpy`` (both ship with ``pip install formatshield[benchmark]``).
    Importing this module does NOT import matplotlib at the module level so that
    the rest of FormatShield can be imported in environments without matplotlib.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_heatmap(
        self,
        results: list[BenchmarkResult],
        output_path: Path | str,
        *,
        title: str = "FormatShield — Accuracy Delta (TTF vs Direct)",
        figsize: tuple[float, float] = (10.0, 5.0),
        dpi: int = 150,
    ) -> Path:
        """
        Write a backend x task accuracy-delta heatmap to *output_path*.

        Each cell shows ``ttf_accuracy - direct_accuracy`` averaged across all
        problems in that (backend, task) pair.  Green cells (positive delta)
        indicate TTF improves accuracy; red cells (negative delta) indicate
        TTF hurts.

        Parameters
        ----------
        results:
            List of :class:`BenchmarkResult` objects from
            :class:`~formatshield.benchmark.harness.BenchmarkHarness`.
        output_path:
            Destination file path. The ``.png`` extension is added automatically
            if not present.
        title:
            Figure title rendered above the heatmap.
        figsize:
            Matplotlib figure size in inches ``(width, height)``.
        dpi:
            Output resolution in dots per inch. 150 dpi is suitable for README
            embedding and paper figures.

        Returns
        -------
        Path
            Absolute path to the written PNG file.
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as exc:
            raise ImportError("pip install formatshield[benchmark] to enable PNG export") from exc

        # Suppress GUI backends in headless environments
        matplotlib.use("Agg")

        out_path = Path(output_path)
        if out_path.suffix.lower() != ".png":
            out_path = out_path.with_suffix(".png")

        # Pivot: collect mean accuracy_delta per (backend, task)
        grid: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in results:
            grid[r.backend][r.task].append(r.accuracy_delta)

        backends = sorted(grid.keys())
        tasks = sorted({r.task for r in results})

        # Build the 2-D matrix (backends as rows, tasks as columns)
        matrix = np.full((len(backends), len(tasks)), fill_value=float("nan"))
        for i, backend in enumerate(backends):
            for j, task in enumerate(tasks):
                deltas = grid[backend].get(task, [])
                if deltas:
                    matrix[i, j] = float(np.mean(deltas))

        # --- Plot ---
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(matrix, cmap=_CMAP, vmin=_VMIN, vmax=_VMAX, aspect="auto")

        # Axis labels
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(tasks, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(backends)))
        ax.set_yticklabels(backends, fontsize=9)

        # Annotate each cell with the numeric value
        for i in range(len(backends)):
            for j in range(len(tasks)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text = f"{val:+.2f}"
                    color = "white" if abs(val) > 0.15 else "black"
                    ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

        fig.colorbar(im, ax=ax, label="Accuracy Delta (TTF - Direct)", pad=0.02)
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_xlabel("Task", fontsize=9)
        ax.set_ylabel("Backend", fontsize=9)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info("PNGExporter: heatmap written to %s", out_path)
        return out_path.resolve()

    def export_latency_bars(
        self,
        results: list[BenchmarkResult],
        output_path: Path | str,
        *,
        figsize: tuple[float, float] = (9.0, 4.5),
        dpi: int = 150,
    ) -> Path:
        """
        Write a grouped bar chart comparing direct vs TTF latency per backend.

        Parameters
        ----------
        results:
            List of :class:`BenchmarkResult` objects.
        output_path:
            Destination file path (``.png`` added if missing).
        figsize:
            Figure dimensions in inches.
        dpi:
            Output resolution.

        Returns
        -------
        Path
            Absolute path to the written PNG file.
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as exc:
            raise ImportError("pip install formatshield[benchmark] to enable PNG export") from exc

        matplotlib.use("Agg")

        out_path = Path(output_path)
        if out_path.suffix.lower() != ".png":
            out_path = out_path.with_suffix(".png")

        # Aggregate mean latencies per backend
        latency_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in results:
            latency_data[r.backend]["direct"].append(r.direct_latency_ms)
            latency_data[r.backend]["ttf"].append(r.ttf_latency_ms)

        backends = sorted(latency_data.keys())
        direct_means = [float(np.mean(latency_data[b]["direct"])) for b in backends]
        ttf_means = [float(np.mean(latency_data[b]["ttf"])) for b in backends]

        x = np.arange(len(backends))
        width = 0.35

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x - width / 2, direct_means, width, label="Direct", color="#4c96d7", alpha=0.85)
        ax.bar(x + width / 2, ttf_means, width, label="TTF (2-pass)", color="#e07b39", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(backends, fontsize=9)
        ax.set_ylabel("Mean latency (ms)", fontsize=9)
        ax.set_title("Direct vs TTF Latency per Backend", fontsize=11, pad=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info("PNGExporter: latency bar chart written to %s", out_path)
        return out_path.resolve()

    def export_threshold_curve(
        self,
        complexity_scores: list[float],
        accuracy_deltas: list[float],
        output_path: Path | str,
        *,
        backend: str = "",
        figsize: tuple[float, float] = (7.0, 4.5),
        dpi: int = 150,
    ) -> Path:
        """
        Write a scatter plot of complexity score vs accuracy delta (Figure 2 in paper).

        Points above the x-axis indicate TTF helped; points below indicate it hurt.
        A vertical dashed line marks the routing threshold.

        Parameters
        ----------
        complexity_scores:
            List of complexity scores from
            :class:`~formatshield.scorer.complexity_scorer.ComplexityScorer`.
        accuracy_deltas:
            Corresponding accuracy deltas (TTF - direct).
        output_path:
            Destination file path.
        backend:
            Backend name for the title.
        figsize:
            Figure dimensions in inches.
        dpi:
            Output resolution.

        Returns
        -------
        Path
            Absolute path to the written PNG file.
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("pip install formatshield[benchmark] to enable PNG export") from exc

        matplotlib.use("Agg")

        out_path = Path(output_path)
        if out_path.suffix.lower() != ".png":
            out_path = out_path.with_suffix(".png")

        colors = ["#27ae60" if d > 0 else "#e74c3c" for d in accuracy_deltas]

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(complexity_scores, accuracy_deltas, c=colors, alpha=0.7, s=40, edgecolors="none")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(
            0.65, color="#7f8c8d", linewidth=1.0, linestyle=":", label="Routing threshold (0.65)"
        )

        ax.set_xlabel("Complexity Score", fontsize=9)
        ax.set_ylabel("Accuracy Delta (TTF - Direct)", fontsize=9)
        title = f"Threshold Curve — {backend}" if backend else "Threshold Curve"
        ax.set_title(title, fontsize=11, pad=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info("PNGExporter: threshold curve written to %s", out_path)
        return out_path.resolve()

    def generate_paper_figures(
        self,
        results: list[BenchmarkResult],
        figures_dir: Path | str,
    ) -> dict[str, Path]:
        """
        Generate all paper figures from a complete benchmark result set.

        Writes:
        * ``heatmap.png`` — Figure 1: accuracy delta grid
        * ``latency_bars.png`` — Figure 3: latency comparison
        * ``threshold_curve.png`` — Figure 2: complexity vs delta scatter

        Parameters
        ----------
        results:
            Full result list from :meth:`~formatshield.benchmark.harness.BenchmarkHarness.run`.
        figures_dir:
            Directory where figures are written.

        Returns
        -------
        dict[str, Path]
            Mapping ``figure_name`` -> absolute path.
        """
        out_dir = Path(figures_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        artifacts: dict[str, Path] = {}

        heatmap_path = self.export_heatmap(results, out_dir / "heatmap.png")
        artifacts["heatmap"] = heatmap_path

        latency_path = self.export_latency_bars(results, out_dir / "latency_bars.png")
        artifacts["latency_bars"] = latency_path

        complexity_scores = [r.complexity_score for r in results]
        accuracy_deltas = [r.accuracy_delta for r in results]
        if complexity_scores:
            threshold_path = self.export_threshold_curve(
                complexity_scores,
                accuracy_deltas,
                out_dir / "threshold_curve.png",
            )
            artifacts["threshold_curve"] = threshold_path

        return artifacts


def _aggregate_by_backend_task(
    results: list[BenchmarkResult],
) -> dict[str, dict[str, Any]]:
    """Return mean accuracy_delta keyed by (backend, task)."""
    raw: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        raw[r.backend][r.task].append(r.accuracy_delta)

    aggregated: dict[str, dict[str, Any]] = {}
    for backend, tasks in raw.items():
        aggregated[backend] = {task: sum(vals) / len(vals) for task, vals in tasks.items()}
    return aggregated
