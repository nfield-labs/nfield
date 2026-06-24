"""Tests for formatshield.io — load document/schema, save/load results."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from formatshield.exceptions import SchemaError
from formatshield.io import load_document, load_results, load_schema, save_results
from formatshield.types import ExtractionResult, ExtractionStatus, Metadata

if TYPE_CHECKING:
    from pathlib import Path


def _result(name: str = "Alice") -> ExtractionResult:
    meta = Metadata(
        K=1,
        K_min=1,
        optimality_gap=0.0,
        quality_score=1.0,
        confidence_level="HIGH",
        fields_extracted=1,
        fields_total=1,
        fields_missing=0,
        fields_conflicted=0,
        fields_needs_revalidation=0,
        per_field_confidence={"name": 0.99},
        retry_rounds=0,
    )
    return ExtractionResult(data={"name": name}, metadata=meta, status=ExtractionStatus.SUCCESS)


class TestLoadDocument:
    def test_reads_utf8_text(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.md"
        p.write_text("# Heading\nbody", encoding="utf-8")
        assert load_document(p) == "# Heading\nbody"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_document(tmp_path / "nope.txt")


class TestLoadSchema:
    def test_parses_json_object(self, tmp_path: Path) -> None:
        p = tmp_path / "schema.json"
        p.write_text(json.dumps({"type": "object", "properties": {}}), encoding="utf-8")
        assert load_schema(p) == {"type": "object", "properties": {}}

    def test_invalid_json_raises_schema_error(self, tmp_path: Path) -> None:
        p = tmp_path / "schema.json"
        p.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(SchemaError, match="not valid JSON"):
            load_schema(p)

    def test_non_object_top_level_raises_schema_error(self, tmp_path: Path) -> None:
        p = tmp_path / "schema.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(SchemaError, match="must contain a JSON object"):
            load_schema(p)


class TestSaveLoadResults:
    def test_round_trip(self, tmp_path: Path) -> None:
        results = [_result("Alice"), _result("Bob")]
        path = tmp_path / "out" / "results.jsonl"  # parent dir does not exist yet
        save_results(results, path)
        loaded = load_results(path)
        assert [r.data["name"] for r in loaded] == ["Alice", "Bob"]
        assert loaded[0].status is ExtractionStatus.SUCCESS
        assert loaded[0].metadata.quality_score == 1.0

    def test_one_json_object_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "results.jsonl"
        save_results([_result(), _result()], path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert all(json.loads(line)["status"] == "success" for line in lines)

    def test_blank_lines_are_skipped_on_load(self, tmp_path: Path) -> None:
        path = tmp_path / "results.jsonl"
        save_results([_result()], path)
        path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
        assert len(load_results(path)) == 1
