"""Published external reference curves for the field-count plot.

These are measured numbers from external papers, transcribed verbatim with their
source, never NField results. They give the field-count plot its comparison
baseline: what a single model call does as the number of requested fields grows.

* IFScale (arXiv:2507.11538) measures instruction-following accuracy as the number
  of simultaneous instructions rises from 10 to 500. It is a different task and a
  different model set than ExtractBench, so it is drawn as a reference for the
  single-call decay *phenomenon*, not as a same-setup head-to-head.
* ExtractBench (arXiv:2602.12247) reports that on its 369-property 10-K/Q schema
  every tested frontier model returns 0% valid whole-document output.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["EXTRACTBENCH_WALL", "IFSCALE_CURVES", "ReferenceCurve", "ReferencePoint"]


@dataclass(frozen=True, slots=True)
class ReferenceCurve:
    """One model's published accuracy-vs-instruction-count decay.

    Args:
        label: Model name as reported by the source.
        source: Citation for the numbers (arXiv id).
        points: ``(instruction_count, accuracy)`` pairs, accuracy in ``[0, 1]``.
    """

    label: str
    source: str
    points: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class ReferencePoint:
    """A single published datapoint, used for a plotted annotation.

    Args:
        label: Human-readable description of what the point marks.
        source: Citation for the number (arXiv id).
        n_fields: Schema field/property count the number was measured at.
        accuracy: Reported accuracy in ``[0, 1]``.
    """

    label: str
    source: str
    n_fields: int
    accuracy: float


# IFScale per-model accuracy at instruction densities 10 / 100 / 250 / 500,
# transcribed from arXiv:2507.11538 (Tables 1-3). The paper's headline: the best
# frontier model reaches only 68% accuracy at 500 instructions. Reasoning models
# (o3, gemini-2.5-pro) hold near-perfect accuracy to a threshold then fall;
# gpt-4.1 / claude-sonnet-4 decay roughly linearly; gpt-4o decays fast.
_IFSCALE_SOURCE = "arXiv:2507.11538"
IFSCALE_CURVES: tuple[ReferenceCurve, ...] = (
    ReferenceCurve(
        "o3 (IFScale)", _IFSCALE_SOURCE, ((10, 1.000), (100, 0.982), (250, 0.978), (500, 0.628))
    ),
    ReferenceCurve(
        "gemini-2.5-pro (IFScale)",
        _IFSCALE_SOURCE,
        ((10, 1.000), (100, 0.984), (250, 0.848), (500, 0.689)),
    ),
    ReferenceCurve(
        "gpt-4.1 (IFScale)",
        _IFSCALE_SOURCE,
        ((10, 0.980), (100, 0.954), (250, 0.740), (500, 0.489)),
    ),
    ReferenceCurve(
        "claude-sonnet-4 (IFScale)",
        _IFSCALE_SOURCE,
        ((10, 1.000), (100, 0.944), (250, 0.772), (500, 0.429)),
    ),
    ReferenceCurve(
        "gpt-4o (IFScale)",
        _IFSCALE_SOURCE,
        ((10, 0.940), (100, 0.490), (250, 0.222), (500, 0.154)),
    ),
)

# ExtractBench's 369-property 10-K/Q schema: every tested frontier model returns
# 0% valid whole-document output (arXiv:2602.12247). Marks the wall single-call
# extraction hits at breadth.
EXTRACTBENCH_WALL = ReferencePoint(
    "ExtractBench 10-K/Q: every frontier model 0% valid",
    "arXiv:2602.12247",
    369,
    0.0,
)
