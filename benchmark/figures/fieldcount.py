"""Field-count curve: NField value accuracy vs schema field count, with references.

Reads the per-document ``scored/*.json`` an ExtractBench run wrote and plots one
point per real document (x = the document's gold field count, y = value accuracy)
against the published single-call decay curves in :mod:`benchmark.figures.reference`.

The story the plot tells is structural, not a same-setup head-to-head: NField's
accuracy stays high as the field count grows into the thousands, while the
reference IFScale curves (a different task and model set) show the accuracy a
single model call loses as the request widens, and the ExtractBench wall marks
where single-call extraction returns 0% valid output. Both scales are honest and
each series carries its own source in the legend.

Plotting is optional: matplotlib is imported only when a plot is requested, so the
loader and text summary run with no extra dependency.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .reference import EXTRACTBENCH_WALL, IFSCALE_CURVES

__all__ = ["FieldCountPoint", "collect_points", "format_summary", "plot_fieldcount_curve"]

# Stable, colorblind-friendly color per ExtractBench domain, so the same domain
# reads the same across regenerations. Domains not listed fall back to grey.
_DOMAIN_COLORS: dict[str, str] = {
    "finance_10kq": "#d62728",
    "finance_credit_agreement": "#ff7f0e",
    "academic_research": "#1f77b4",
    "hiring_resume": "#2ca02c",
    "sport_swimming": "#9467bd",
}
_DEFAULT_COLOR = "#7f7f7f"

# Human-readable domain labels for the legend.
_DOMAIN_LABELS: dict[str, str] = {
    "finance_10kq": "finance / 10-K,Q",
    "finance_credit_agreement": "finance / credit",
    "academic_research": "academic",
    "hiring_resume": "resume",
    "sport_swimming": "swimming",
}

# Reference decay curves are drawn in shades of grey so the vivid domain colors
# read as "NField" and the greys read as "single-call baseline". One grey per
# IFScale model, dark to light, kept distinct from the domain palette above.
_REFERENCE_GREYS: tuple[str, ...] = ("#111111", "#3f3f3f", "#5f5f5f", "#7f7f7f", "#a5a5a5")

# The field count above which we report a separate mean, to show the curve does
# not decay past the range the reference curves stop at (IFScale ends at 500).
_WIDE_SCHEMA_THRESHOLD: int = 500


@dataclass(frozen=True, slots=True)
class FieldCountPoint:
    """One document's measured result: its field count and value accuracy.

    Args:
        document: Source document name.
        domain: ExtractBench domain (directory name), used for color/label.
        n_fields: The document's gold field count (the x-axis, N).
        value_accuracy: Strict per-field value accuracy in ``[0, 1]``.
        value_accuracy_judged: Accuracy after the benchmark's LLM judge re-scores
            the tiers it marks for judging (falls back to the strict value).
    """

    document: str
    domain: str
    n_fields: int
    value_accuracy: float
    value_accuracy_judged: float


def collect_points(result_dir: Path) -> list[FieldCountPoint]:
    """Read every ``scored/*.json`` under ``result_dir`` into field-count points.

    Args:
        result_dir: An ExtractBench result directory (handles a ``native/``
            budget subfolder or a flat layout).

    Returns:
        Points sorted by ``n_fields`` then document name.

    Raises:
        FileNotFoundError: If no ``scored/`` outputs are found under ``result_dir``.
    """
    scored = sorted(result_dir.glob("**/scored/*.json"))
    if not scored:
        raise FileNotFoundError(f"no scored/ outputs under {result_dir}")

    points = [_point(path) for path in scored]
    points.sort(key=lambda p: (p.n_fields, p.document))
    return points


def _point(path: Path) -> FieldCountPoint:
    record = json.loads(path.read_text(encoding="utf-8"))
    strict = float(record["value_accuracy"])
    return FieldCountPoint(
        document=str(record.get("document", path.stem)),
        domain=path.parent.parent.name,
        n_fields=int(record["gold_fields"]),
        value_accuracy=strict,
        value_accuracy_judged=float(record.get("value_accuracy_judged", strict)),
    )


def format_summary(points: list[FieldCountPoint], *, judged: bool = False) -> str:
    """Return a short text summary showing accuracy holds as the field count grows.

    Args:
        points: The collected points.
        judged: Report the judged accuracy instead of the strict one.

    Returns:
        A few lines: point count, N range, and mean accuracy below vs. at-or-above
        the wide-schema threshold (so any decay with N would show).
    """
    if not points:
        return "no points"
    accuracy = _accuracy_getter(judged=judged)
    narrow = [accuracy(p) for p in points if p.n_fields < _WIDE_SCHEMA_THRESHOLD]
    wide = [accuracy(p) for p in points if p.n_fields >= _WIDE_SCHEMA_THRESHOLD]
    lines = [
        f"points        : {len(points)}",
        f"N range       : {points[0].n_fields} -> {points[-1].n_fields} fields",
        f"mean acc <{_WIDE_SCHEMA_THRESHOLD}   : {_mean(narrow):.3f}  ({len(narrow)} docs)",
        f"mean acc >={_WIDE_SCHEMA_THRESHOLD}  : {_mean(wide):.3f}  ({len(wide)} docs)",
    ]
    return "\n".join(lines)


def plot_fieldcount_curve(
    points: list[FieldCountPoint], path: Path, *, judged: bool = False
) -> Path | None:
    """Plot NField accuracy vs field count against the reference decay curves.

    Args:
        points: Collected NField per-document points.
        path: Output image path (parent directories are created).
        judged: Plot the judged accuracy instead of the strict one.

    Returns:
        The written path, or ``None`` if matplotlib is unavailable or there are
        no points.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not points:
        return None

    accuracy = _accuracy_getter(judged=judged)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 6))

    # Reference single-call decay curves (dashed, grey) drawn first so NField's
    # points sit on top.
    for index, curve in enumerate(IFSCALE_CURVES):
        axis.plot(
            [n for n, _ in curve.points],
            [a for _, a in curve.points],
            linestyle="--",
            linewidth=1.2,
            color=_REFERENCE_GREYS[index % len(_REFERENCE_GREYS)],
            alpha=0.7,
            marker=".",
            label=curve.label,
        )
    axis.scatter(
        [EXTRACTBENCH_WALL.n_fields],
        [EXTRACTBENCH_WALL.accuracy],
        marker="x",
        s=90,
        color="black",
        zorder=5,
        label=EXTRACTBENCH_WALL.label,
    )

    # NField measured points, one marker per document, colored by domain.
    for domain in sorted({p.domain for p in points}):
        domain_points = [p for p in points if p.domain == domain]
        axis.scatter(
            [p.n_fields for p in domain_points],
            [accuracy(p) for p in domain_points],
            s=48,
            color=_DOMAIN_COLORS.get(domain, _DEFAULT_COLOR),
            edgecolors="white",
            linewidths=0.5,
            zorder=6,
            label=f"NField - {_DOMAIN_LABELS.get(domain, domain)}",
        )

    axis.set_xscale("log")
    axis.set_xlabel("N (document field count)")
    axis.set_ylabel("value accuracy (judged)" if judged else "value accuracy (strict)")
    axis.set_ylim(0.0, 1.02)
    axis.grid(True, which="both", linestyle=":", alpha=0.4)
    axis.legend(fontsize=7, loc="lower left", ncol=2)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def _accuracy_getter(*, judged: bool):  # type: ignore[no-untyped-def]
    return (lambda p: p.value_accuracy_judged) if judged else (lambda p: p.value_accuracy)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m benchmark.figures.fieldcount <result_dir>``."""
    parser = argparse.ArgumentParser(prog="benchmark.figures.fieldcount", description=__doc__)
    parser.add_argument("result_dir", type=Path, help="an ExtractBench result directory")
    parser.add_argument("--judged", action="store_true", help="use judged accuracy")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output image path (default: <result_dir>/ifscale/fieldcount.png)",
    )
    args = parser.parse_args(argv)

    points = collect_points(args.result_dir)
    print(format_summary(points, judged=args.judged))

    out = args.out or args.result_dir / "ifscale" / "fieldcount.png"
    plotted = plot_fieldcount_curve(points, out, judged=args.judged)
    print(f"\nplot -> {plotted}" if plotted else "\nplot -> skipped (matplotlib absent)")


if __name__ == "__main__":
    main()
