"""Tests for the field-count point loader and summary."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from benchmark.fieldcount import (
    FieldCountPoint,
    collect_points,
    format_summary,
    plot_fieldcount_curve,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_scored(run: Path, domain: str, doc: str, payload: dict) -> None:
    scored = run / "native" / domain / "scored"
    scored.mkdir(parents=True, exist_ok=True)
    (scored / f"{doc}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_collect_points_reads_every_scored_file(tmp_path: Path) -> None:
    _write_scored(
        tmp_path,
        "finance_credit_agreement",
        "small",
        {"document": "small", "gold_fields": 13, "value_accuracy": 0.6},
    )
    _write_scored(
        tmp_path,
        "finance_10kq",
        "wide",
        {
            "document": "wide",
            "gold_fields": 1407,
            "value_accuracy": 0.76,
            "value_accuracy_judged": 0.82,
        },
    )

    points = collect_points(tmp_path)

    assert [p.n_fields for p in points] == [13, 1407]  # sorted by N
    assert points[0].domain == "finance_credit_agreement"
    assert points[1].value_accuracy_judged == 0.82


def test_collect_points_defaults_judged_to_strict(tmp_path: Path) -> None:
    _write_scored(tmp_path, "sport_swimming", "t1", {"gold_fields": 100, "value_accuracy": 0.9})
    (point,) = collect_points(tmp_path)
    assert point.value_accuracy_judged == 0.9


def test_collect_points_raises_when_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        collect_points(tmp_path)


def test_format_summary_splits_on_the_wide_schema_threshold() -> None:
    points = [
        FieldCountPoint("a", "d", 100, 0.90, 0.92),
        FieldCountPoint("b", "d", 1000, 0.80, 0.85),
    ]
    summary = format_summary(points)
    assert "N range       : 100 -> 1000 fields" in summary
    assert "0.900" in summary  # mean below threshold
    assert "0.800" in summary  # mean at or above threshold


def test_plot_returns_none_without_points(tmp_path: Path) -> None:
    assert plot_fieldcount_curve([], tmp_path / "out.png") is None


def test_plot_writes_a_file_when_matplotlib_available(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    points = [FieldCountPoint("a", "finance_10kq", 1407, 0.76, 0.82)]
    out = plot_fieldcount_curve(points, tmp_path / "plots" / "fieldcount.png")
    assert out is not None
    assert out.exists()
