"""nfield on the wide-schema sweep: coverage by field count, and call-count scaling.

Two figures in the shared benchmark style, both reading a runner's ``summary.csv``:

- ``plot_coverage`` - coverage per fixture (share of schema fields filled, not left
  empty), fixtures ordered by field count. Coverage is the system's job: nfield
  decomposes a wide schema into bounded calls so every field gets asked, where a
  single call truncates and drops the tail. Value accuracy is the model's job and is
  not what this compares.
- ``plot_scale`` - nfield's call count against the computed minimum as the field
  count climbs into the thousands, showing the split stays near optimal with no storm.

    uv run python -m benchmark.figures.charts <run_dir>
    uv run python -m benchmark.figures.charts <scale_run_dir> --scale
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import _figstyle

# Baselines cycle through these muted tones; nfield is always the deep blue below.
_BASELINE_PALETTE: tuple[str, ...] = ("#c44e52", "#6ba3d6", "#e0894f", "#8d9db6", "#6d9f71")
_NFIELD = "nfield"
# The fair comparison hands every method the same budget; native lets each take its
# provider maximum, which flatters a single call. Prefer the equal one.
_FAIR_BUDGET = "constrained"


def _load_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "summary.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _model_of(run_dir: Path) -> str:
    manifest = run_dir / "MANIFEST.json"
    if manifest.exists():
        model = json.loads(manifest.read_text(encoding="utf-8")).get("model")
        if isinstance(model, str):
            return model
    return "model"


def _by_fixture(
    rows: list[dict[str, str]], column: str
) -> tuple[list[str], list[int], list[str], dict[str, dict[str, float]]]:
    """Return (fixtures ordered by field count, sizes, methods, value[method][fixture])."""
    sizes: dict[str, int] = {}
    values: dict[str, dict[str, float]] = {}
    methods: list[str] = []
    for r in rows:
        if r["budget"] != _FAIR_BUDGET or not r[column]:
            continue
        method, fixture = r["method"], r["fixture"]
        sizes[fixture] = int(r["n_fields"])
        values.setdefault(method, {})[fixture] = float(r[column])
        if method not in methods:
            methods.append(method)
    fixtures = sorted(sizes, key=lambda fx: sizes[fx])
    ordered = ([_NFIELD] if _NFIELD in methods else []) + sorted(
        m for m in methods if m != _NFIELD
    )
    return fixtures, [sizes[fx] for fx in fixtures], ordered, values


def plot_coverage(run_dir: Path, out: Path) -> Path | None:
    """Draw coverage per fixture (nfield vs baselines), fixtures ordered by field count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_rows(run_dir)
    fixtures, fields, methods, values = _by_fixture(rows, "coverage")
    if not fixtures or _NFIELD not in methods:
        return None

    baselines = [m for m in methods if m != _NFIELD]
    colors = {m: _BASELINE_PALETTE[i % len(_BASELINE_PALETTE)] for i, m in enumerate(baselines)}
    colors[_NFIELD] = _figstyle.NFIELD

    _figstyle.apply_rcparams()
    fig, ax = plt.subplots(figsize=(11, 6))
    _figstyle.style_axes(ax)

    group = list(range(len(fixtures)))
    width = min(0.8 / len(methods), 0.16)
    offset0 = -width * (len(methods) - 1) / 2
    for i, method in enumerate(methods):
        heights = [values[method].get(fx, 0.0) for fx in fixtures]
        bars = ax.bar(
            [g + offset0 + i * width for g in group],
            heights,
            width,
            color=colors[method],
            label=method,
            zorder=3,
            edgecolor="white",
            linewidth=0.5,
        )
        if method == _NFIELD:
            ax.bar_label(
                bars,
                labels=[f"{h * 100:.0f}" for h in heights],
                padding=2,
                fontsize=8.5,
                color=_figstyle.NFIELD,
                fontweight="bold",
            )

    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xticks(group)
    ax.set_xticklabels(
        [f"{fx}\n{n} fields" for fx, n in zip(fixtures, fields, strict=True)], fontsize=9.5
    )
    ax.set_ylabel("coverage (fields filled, not left empty)", fontsize=10.5)
    _figstyle.title_block(
        ax,
        "nfield fills the whole schema as it widens; single-call drops fields",
        f"groq/{_model_of(run_dir).split('/')[-1]}  ·  equal token budget  ·  "
        "coverage is the system's job, not the model's",
    )
    ax.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=len(methods), handlelength=1.3)
    _figstyle.caption(
        fig,
        "Same document, schema, budget, and scorer for every method. Decomposing the schema keeps "
        "every field within a call that fits, so coverage stays high where a single call truncates "
        "and leaves the tail empty.",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_scale(run_dir: Path, out: Path) -> Path | None:
    """Draw nfield's call count vs the computed minimum as the field count climbs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        r
        for r in _load_rows(run_dir)
        if r["method"] == _NFIELD and r["budget"] == _FAIR_BUDGET and r["k"]
    ]
    if not rows:
        return None
    rows.sort(key=lambda r: int(r["n_fields"]))
    fields = [int(r["n_fields"]) for r in rows]
    k = [int(r["k"]) for r in rows]
    k_min = [int(r["k_min"]) for r in rows]
    coverage = [float(r["coverage"]) for r in rows]

    _figstyle.apply_rcparams()
    fig, ax = plt.subplots(figsize=(11, 6))
    _figstyle.style_axes(ax, grid_axis="both")

    ax.plot(
        fields,
        k_min,
        marker="o",
        color=_figstyle.TEXT_MUTED,
        linewidth=1.6,
        linestyle="--",
        label="computed minimum calls",
        zorder=3,
    )
    ax.plot(
        fields,
        k,
        marker="o",
        color=_figstyle.NFIELD,
        linewidth=2.2,
        label="nfield calls",
        zorder=4,
    )
    for x, y, cov in zip(fields, k, coverage, strict=True):
        ax.annotate(
            f"{y} calls\n{cov * 100:.0f}% filled",
            (x, y),
            textcoords="offset points",
            xytext=(0, 12),
            fontsize=8.5,
            color=_figstyle.NFIELD,
            fontweight="bold",
            ha="center",
        )

    ax.set_xlabel("schema fields", fontsize=10.5)
    ax.set_ylabel("model calls", fontsize=10.5)
    ax.set_xlim(0, max(fields) * 1.1)
    ax.set_ylim(0, max(k) * 1.25)
    _figstyle.title_block(
        ax,
        "nfield scales to thousands of fields with no call storm",
        f"groq/{_model_of(run_dir).split('/')[-1]}  ·  calls stay within a couple of the "
        "minimum the schema needs",
    )
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", handlelength=1.8)
    _figstyle.caption(
        fig,
        "The schema is split exactly as much as the budget requires, so the number of calls tracks "
        "the computed minimum and coverage holds near 100% out to 5,641 fields.",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI: draw the coverage comparison, or the scale plot with ``--scale``."""
    parser = argparse.ArgumentParser(prog="benchmark.figures.charts", description=__doc__)
    parser.add_argument("run_dir", type=Path, help="a completed runner result directory")
    parser.add_argument("--scale", action="store_true", help="draw the call-count scale plot")
    parser.add_argument("--out", type=Path, default=None, help="output image path")
    args = parser.parse_args(argv)
    if args.scale:
        out = args.out or args.run_dir.parent / "figures" / "scale_no_call_storm.png"
        written = plot_scale(args.run_dir, out)
    else:
        out = args.out or args.run_dir.parent / "figures" / "coverage_by_fieldcount.png"
        written = plot_coverage(args.run_dir, out)
    print(f"wrote {written}" if written else "skipped (no matching nfield rows)")


if __name__ == "__main__":
    main()
