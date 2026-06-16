"""Adversarial benchmark tests — fairness, coverage capping, instructor converter.

* the gold-based coverage recorded by the runner must never exceed 1.0, even
  when an adapter over-extracts;
* the instructor schema->Pydantic converter must build an all-optional model
  (partial extraction is not rejected), and a digit-leading schema key must
  round-trip through a Field alias so gold path-alignment is preserved.
"""

from __future__ import annotations

from benchmark.adapters import AdapterOutput
from benchmark.adapters.instructor_adapter import (
    _annotation,
    _model_from_schema,
    _safe_field,
)
from benchmark.datasets import LoadedDataset
from benchmark.runner import _record, run_sweep
from benchmark.score import score


def _dataset(gold, schema):
    return LoadedDataset(name="toy", schema=schema, document="doc", gold=gold)


class TestCoverageCappedAtOne:
    """The runner's recorded coverage is gold-based and never exceeds 1.0."""

    def test_over_extraction_does_not_push_coverage_above_one(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        gold = {"a": "x"}
        # Adapter reports MORE leaves than the gold has (model invented keys):
        # raw fields_extracted/fields_total would be 3/1 = 3.0; the gold-based
        # report must clamp it.
        output = AdapterOutput(
            data={"a": "x", "b": "y", "c": "z"},
            fields_total=1,
            fields_extracted=3,
            k=1,
            k_min=1,
        )
        report = score(output.data, gold, schema, call_failed=0)
        record = _record("m", "toy", 0, output, report)
        assert record["coverage"] <= 1.0

    def test_coverage_only_fixture_falls_back_to_raw_ratio(self):
        # No gold -> no report -> raw ratio (this is the only path that may
        # exceed 1.0, and it is explicitly the coverage-only fallback).
        output = AdapterOutput(data={}, fields_total=2, fields_extracted=1, k=1, k_min=1)
        record = _record("m", "toy", 0, output, None)
        assert record["coverage"] == 0.5

    def test_full_sweep_record_coverage_within_unit_interval(self, tmp_path):
        from dataclasses import dataclass

        schema = {"type": "object", "properties": {"a": {"type": "string"}}}

        @dataclass
        class _Over:
            name: str = "over"

            def run(
                self,
                document,
                schema,
                *,
                model,
                context_window,
                max_output_tokens,
                instructions="",
            ):
                return AdapterOutput(
                    data={"a": "x", "extra": "y"},
                    fields_total=1,
                    fields_extracted=2,
                    k=1,
                    k_min=1,
                )

        artifacts = run_sweep(
            _Over(),
            _dataset({"a": "x"}, schema),
            model="groq/x",
            seeds=1,
            out_dir=tmp_path / "o",
            context_window=8192,
            max_output_tokens=2048,
            budget="native",
        )
        import json

        record = json.loads(artifacts.raw_path.read_text(encoding="utf-8"))[0]
        assert record["coverage"] <= 1.0


class TestInstructorConverter:
    def test_model_is_all_optional(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "n": {"type": "integer"}},
        }
        model = _model_from_schema(schema, "T", depth=0)
        assert all(f.default is None for f in model.model_fields.values())

    def test_enum_maps_to_str(self):
        assert _annotation({"enum": ["X", "Y"]}, "e", 0) is str

    def test_nested_object_becomes_submodel(self):
        schema = {
            "type": "object",
            "properties": {"o": {"type": "object", "properties": {"x": {"type": "string"}}}},
        }
        model = _model_from_schema(schema, "T", depth=0)
        sub = model.model_fields["o"].annotation
        # Optional[submodel]; the submodel carries the nested field.
        assert "x" in getattr(_unwrap_optional(sub), "model_fields", {})

    def test_digit_leading_key_round_trips_via_alias(self):
        # A real fixture key like "0_14_years" is not a valid identifier, so the
        # converter stores it as field_0 but keeps the original key as a Field
        # alias. With populate_by_name + dump-by-alias the round trip restores the
        # gold key, so path alignment is preserved.
        assert _safe_field("0_14_years", 0) == "field_0"
        model = _model_from_schema(
            {"type": "object", "properties": {"0_14_years": {"type": "string"}}}, "T", depth=0
        )
        assert "field_0" in model.model_fields
        instance = model.model_validate({"0_14_years": "50%"})
        assert instance.model_dump(by_alias=True) == {"0_14_years": "50%"}


def _unwrap_optional(annotation):
    import typing

    args = typing.get_args(annotation)
    return next((a for a in args if a is not type(None)), annotation)
