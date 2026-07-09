"""Shared palette and styling for the benchmark figures, so the charts match.

One place fixes the look of every plot: nfield in deep blue, the single-call /
baseline series in a muted brick, a light-steel accent for a second model, and a
common clean frame (soft grid behind the data, no top/right spines, left-aligned
bold title with a muted subtitle).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Series colours.
NFIELD = "#11497e"  # deep blue - nfield
BASELINE = "#c44e52"  # muted brick - single-call / frontier baseline
ACCENT = "#6ba3d6"  # light steel - a second model or judged series
GRID = "#e6e6e6"
TEXT_MUTED = "#5b6570"

# Ordered palette for charts that colour by model (light -> deep, then brick).
MODEL_PALETTE: tuple[str, ...] = (ACCENT, NFIELD, BASELINE, "#e0894f")


def apply_rcparams() -> None:
    """Set the shared font and axis edge colour on the global rcParams."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#c9ced6"})


def style_axes(ax: Axes, *, grid_axis: Literal["both", "x", "y"] = "y") -> None:
    """Apply the common frame: soft grid behind the data, no top/right spines."""
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def title_block(ax: Axes, title: str, subtitle: str) -> None:
    """Draw a left-aligned bold title with a muted subtitle above the axes."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=34, loc="left")
    ax.text(0, 1.045, subtitle, transform=ax.transAxes, fontsize=9.5, color=TEXT_MUTED)


def caption(fig: Any, text: str, *, y: float = -0.02) -> None:
    """Draw a small muted caption centered under the figure."""
    fig.text(0.5, y, text, ha="center", fontsize=8, color=TEXT_MUTED)
