"""Report aggregator tests — JSONL collection, CSV, and table rendering."""

from __future__ import annotations

import csv
import json

from benchmark.report import collect_rows, format_table, write_summary_csv


def _write_sidecar(result_dir, name: str, records: list[dict]) -> None:
    raw = result_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{name}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def _record(
    method: str, fixture: str, seed: int, *, va: float | None, n: int, budget: str = ""
) -> dict:
    record = {
        "method": method,
        "fixture": fixture,
        "budget": budget,
        "seed": seed,
        "fields_total": n,
        "coverage": 0.9,
        "k": 5,
        "k_min": 4,
        "optimality_gap": 0.25,
        "elapsed_seconds": 12.0,
        "error": None,
    }
    if va is not None:
        record["value_accuracy"] = va
    return record


def test_collect_rows_reads_and_sorts(tmp_path):
    _write_sidecar(
        tmp_path,
        "nfield_clinicaltrial",
        [
            _record("nfield", "clinicaltrial", 1, va=0.8, n=304),
            _record("nfield", "clinicaltrial", 0, va=0.9, n=304),
        ],
    )
    rows = collect_rows(tmp_path)
    assert [r.seed for r in rows] == [0, 1]  # sorted by seed within the cell
    assert rows[0].value_accuracy == 0.9
    assert rows[0].n_fields == 304


def test_collect_rows_spans_budget_subfolders(tmp_path):
    # The sweep layout: one run dir with per-budget subfolders, each holding raw/.
    for budget in ("native", "constrained"):
        raw = tmp_path / budget / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        (raw / "nfield_x.json").write_text(
            json.dumps([_record("nfield", "x", 0, va=0.5, n=10, budget=budget)]),
            encoding="utf-8",
        )
    rows = collect_rows(tmp_path)
    assert {r.budget for r in rows} == {"native", "constrained"}
    # Sorted by budget first, so constrained precedes native.
    assert rows[0].budget == "constrained"

    csv_path = tmp_path / "summary.csv"
    write_summary_csv(rows, csv_path)
    with csv_path.open(encoding="utf-8") as handle:
        loaded = list(csv.DictReader(handle))
    assert {r["budget"] for r in loaded} == {"native", "constrained"}


def test_coverage_only_row_has_no_value_accuracy(tmp_path):
    _write_sidecar(tmp_path, "nfield_war", [_record("nfield", "war_and_peace", 0, va=None, n=200)])
    rows = collect_rows(tmp_path)
    assert rows[0].value_accuracy is None


def test_summary_csv_round_trips(tmp_path):
    _write_sidecar(tmp_path, "nfield_x", [_record("nfield", "x", 0, va=0.5, n=10)])
    rows = collect_rows(tmp_path)
    csv_path = tmp_path / "summary.csv"
    write_summary_csv(rows, csv_path)
    with csv_path.open(encoding="utf-8") as handle:
        loaded = list(csv.DictReader(handle))
    assert loaded[0]["method"] == "nfield"
    assert loaded[0]["value_accuracy"] == "0.5"
    assert loaded[0]["n_fields"] == "10"


def test_format_table_marks_missing_va_with_dash(tmp_path):
    _write_sidecar(tmp_path, "nfield_war", [_record("nfield", "war_and_peace", 0, va=None, n=200)])
    table = format_table(collect_rows(tmp_path))
    assert "war_and_peace" in table
    assert " - " in table or table.endswith("-") or "-" in table.split("\n")[2]
