"""nfield vs frontier single-call on ExtractBench, per domain.

Reads a completed ExtractBench sweep's ``native/summary.csv`` (one row per domain)
and draws nfield's per-field accuracy next to the frontier single-call pass rate the
ExtractBench paper reports for the same domain and the same metric.

Both numbers are the *same* measure: the fraction of gold fields whose value passes
the threshold declared in the schema's ``evaluation_config`` (an LLM judge on the
string/array tiers, deterministic elsewhere). The paper's frontier pass rate and
nfield's judged value accuracy use that identical config, so the bars are directly
comparable. nfield runs on one open model (qwen3.6-27b); the frontier numbers are the
best of six flagship models asked for the whole schema in one call - so the gap is
the extraction strategy, not the model.

    uv run python -m benchmark.extractbench_domains <result_dir>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from . import _figstyle

_PAPER_CITATION: str = "ExtractBench, arXiv:2602.12247"
# Per-domain frontier single-call pass rate and the aggregate, quoted from the paper
# (best-of-six across Gemini 3 Pro/Flash, GPT-5/5.2, Claude Opus/Sonnet 4.5; aggregate
# 4.6% = 844/18,516 field evaluations; best single model Gemini 3 Flash 6.9%).
_PAPER_FRONTIER_PASS: dict[str, float] = {
    "sport_swimming": 0.125,
    "academic_research": 0.208,
    "finance_credit_agreement": 0.563,
    "hiring_resume": 0.184,
    "finance_10kq": 0.0,
}
_PAPER_AGG_PASS_RATE: float = 0.046

# Dataset key -> readable axis label, in display order.
_DOMAIN_LABELS: dict[str, str] = {
    "sport_swimming": "swimming",
    "academic_research": "academic",
    "finance_credit_agreement": "credit",
    "hiring_resume": "resume",
    "finance_10kq": "10-Q\n(369 fields)",
}

_NFIELD_COLOR = _figstyle.NFIELD
_FRONTIER_COLOR = _figstyle.BASELINE
_TEXT_MUTED = _figstyle.TEXT_MUTED


def load_domains(result_dir: Path) -> list[dict[str, float | str]]:
    """Read per-domain rows (nfield strict+judged and the paper's frontier pass rate)."""
    summary = result_dir / "native" / "summary.csv"
    with summary.open(encoding="utf-8") as fh:
        rows = {r["dataset"]: r for r in csv.DictReader(fh)}
    ordered: list[dict[str, float | str]] = []
    for key, label in _DOMAIN_LABELS.items():
        if key not in rows:
            continue
        r = rows[key]
        ordered.append(
            {
                "label": label,
                "gold": int(r["gold_fields"]),
                "strict": float(r["value_accuracy"]),
                "judged": float(r.get("value_accuracy_judged", r["value_accuracy"])),
                "frontier": _PAPER_FRONTIER_PASS.get(key, 0.0),
            }
        )
    return ordered


def _weighted(domains: list[dict[str, float | str]], key: str) -> float:
    """Field-weighted mean of ``key`` over the domains (each gold field counts once)."""
    total = sum(int(d["gold"]) for d in domains)
    if not total:
        return 0.0
    return sum(float(d[key]) * int(d["gold"]) for d in domains) / total


def plot_domains(result_dir: Path, out: Path) -> Path:
    """Draw nfield vs frontier single-call per domain, save to ``out``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    domains = load_domains(result_dir)
    nfield_overall = _weighted(domains, "judged")
    strict_overall = _weighted(domains, "strict")
    total_gold = sum(int(d["gold"]) for d in domains)
    labels = [str(d["label"]) for d in domains] + ["overall"]
    nfield = [float(d["judged"]) for d in domains] + [nfield_overall]
    frontier = [float(d["frontier"]) for d in domains] + [_PAPER_AGG_PASS_RATE]

    _figstyle.apply_rcparams()
    fig, ax = plt.subplots(figsize=(11, 6))
    x = list(range(len(labels)))
    width = 0.4
    ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, color="#f2f5f8", zorder=0)
    _figstyle.style_axes(ax)

    bn = ax.bar(
        [i - width / 2 for i in x],
        nfield,
        width,
        color=_NFIELD_COLOR,
        label="nfield (qwen3.6-27b)",
        zorder=3,
    )
    bf = ax.bar(
        [i + width / 2 for i in x],
        frontier,
        width,
        color=_FRONTIER_COLOR,
        label="frontier single-call (best of 6)",
        zorder=3,
    )
    ax.bar_label(
        bn,
        fmt="%.0f%%",
        labels=[f"{v * 100:.0f}%" for v in nfield],
        padding=2,
        fontsize=9,
        color=_NFIELD_COLOR,
        fontweight="bold",
    )
    ax.bar_label(
        bf,
        labels=[f"{v * 100:.0f}%" for v in frontier],
        padding=2,
        fontsize=9,
        color=_FRONTIER_COLOR,
    )

    ax.set_ylabel("fields passing the schema's own metric", fontsize=10.5)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(
        "nfield holds where single-call extraction collapses",
        fontsize=14,
        fontweight="bold",
        pad=34,
        loc="left",
    )
    ax.text(
        0,
        1.045,
        f"{_PAPER_CITATION}, same per-field metric  ·  nfield {nfield_overall:.0%} "
        f"(strict floor {strict_overall:.0%}) vs frontier {_PAPER_AGG_PASS_RATE:.0%}, "
        f"across {total_gold:,} gold fields",
        transform=ax.transAxes,
        fontsize=9.5,
        color=_TEXT_MUTED,
    )
    ax.legend(frameon=False, fontsize=10, loc="upper right", ncol=2, handlelength=1.4)
    _figstyle.caption(
        fig,
        "Frontier = best of Gemini 3 Pro/Flash, GPT-5/5.2, Claude Opus/Sonnet 4.5 asked for "
        "the whole schema in one call. nfield decomposes it; both scored on the identical "
        "evaluation_config, so the gap is the method, not the model.",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI: draw the comparison into ``benchmark/results/extractbench/figures/``."""
    parser = argparse.ArgumentParser(prog="benchmark.extractbench_domains", description=__doc__)
    parser.add_argument("result_dir", type=Path, help="a completed ExtractBench sweep directory")
    parser.add_argument("--out", type=Path, default=None, help="output image path")
    args = parser.parse_args(argv)
    out = args.out or args.result_dir.parent / "figures" / "nfield_vs_frontier_by_domain.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plotted = plot_domains(args.result_dir, out)
    domains = load_domains(args.result_dir)
    print(
        f"nfield judged {_weighted(domains, 'judged'):.3f}  strict {_weighted(domains, 'strict'):.3f}"
        f"  frontier {_PAPER_AGG_PASS_RATE:.3f}"
    )
    print(f"wrote {plotted}")


if __name__ == "__main__":
    main()
