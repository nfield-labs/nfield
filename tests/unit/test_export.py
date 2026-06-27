"""Tests for nfield.export - DataFrame / CSV export (optional pandas)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nfield.export import result_to_dataframe, results_to_csv, results_to_dataframe
from nfield.types import ExtractionResult, ExtractionStatus, FieldResult, Metadata

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("pandas")


def _meta() -> Metadata:
    return Metadata(
        K=1,
        K_min=1,
        optimality_gap=0.0,
        quality_score=0.9,
        confidence_level="HIGH",
        fields_extracted=2,
        fields_total=2,
        fields_missing=0,
        fields_conflicted=0,
        fields_needs_revalidation=0,
        per_field_confidence={},
        retry_rounds=0,
    )


def _result(name: str, age: int) -> ExtractionResult:
    fields = (
        FieldResult(path="name", value=name, confidence=0.99),
        FieldResult(path="age", value=age, confidence=0.99),
    )
    return ExtractionResult(
        data={"name": name, "age": age},
        metadata=_meta(),
        status=ExtractionStatus.SUCCESS,
        fields=fields,
    )


class TestResultsToDataFrame:
    def test_one_row_per_result_columns_are_field_paths(self) -> None:
        df = results_to_dataframe([_result("Alice", 30), _result("Bob", 41)])
        assert list(df.columns) == ["name", "age"]
        assert len(df) == 2
        assert df.iloc[1]["name"] == "Bob"

    def test_single_result_is_one_row(self) -> None:
        df = result_to_dataframe(_result("Alice", 30))
        assert len(df) == 1
        assert df.iloc[0]["age"] == 30

    def test_include_metadata_adds_prefixed_columns(self) -> None:
        df = results_to_dataframe([_result("Alice", 30)], include_metadata=True)
        assert "_meta.status" in df.columns
        assert df.iloc[0]["_meta.status"] == "success"
        assert df.iloc[0]["_meta.quality_score"] == 0.9

    def test_falls_back_to_data_when_no_field_results(self) -> None:
        bare = ExtractionResult(
            data={"name": "Zed"}, metadata=_meta(), status=ExtractionStatus.SUCCESS
        )
        df = results_to_dataframe([bare])
        assert df.iloc[0]["name"] == "Zed"


class TestResultsToCsv:
    def test_writes_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "out.csv"
        results_to_csv([_result("Alice", 30), _result("Bob", 41)], path)
        text = path.read_text(encoding="utf-8")
        assert "name,age" in text.replace(" ", "")
        assert "Bob" in text
